# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import random

import numpy as np
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from PIL import Image
from safetensors.torch import load_file

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer


SINGLE_IMAGE_PROMPT = "Analyze the perspective and projection realism of this image."
PAIR_PROMPT = "Compare the two images and explain the perspective and projection realism difference."
LORA_VARIANTS = {
    "normal": {
        "checkpoint_path": (
            "/data/bagel/repo/Bagel/results/reason_heatmap_lora/"
            "checkpoints/0006200"
        ),
        "output_dir": "./results/reason_heatmap_lora/inference",
    },
    "mse": {
        "checkpoint_path": (
            "/data/bagel/repo/Bagel/results/reason_heatmap_lora_mse_4gpu/"
            "checkpoints/0006000"
        ),
        "output_dir": "./results/reason_heatmap_lora_mse_4gpu/inference",
    },
}


class DeviceImageTransform:
    def __init__(self, transform, device):
        self.transform = transform
        self.device = device
        self.resize_transform = transform.resize_transform

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs).to(self.device)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run single-GPU teacher-forced inference with a reason heatmap LoRA."
    )
    parser.add_argument(
        "--model_path",
        default="/data/bagel/repo/agent/bpipe/models/BAGEL-7B-MoT",
    )
    parser.add_argument(
        "--lora_variant",
        choices=tuple(LORA_VARIANTS),
        default="normal",
        help="Use the normal LoRA by default, or select the MSE-only LoRA.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
    )
    parser.add_argument(
        "--data_dir",
        default="/data/bagel/data/perspective_5k/canonical_5k_clean_4402",
    )
    parser.add_argument("--metadata_path", default=None)
    parser.add_argument("--row_index", type=int, default=0)
    parser.add_argument(
        "--sample_type",
        choices=("good", "bad", "pair"),
        default="bad",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--cfg_text_scale", type=float, default=1.0)
    parser.add_argument("--cfg_img_scale", type=float, default=2.0)
    args = parser.parse_args()
    variant = LORA_VARIANTS[args.lora_variant]
    args.checkpoint_path = args.checkpoint_path or variant["checkpoint_path"]
    args.output_dir = args.output_dir or variant["output_dir"]
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl_row(path, row_index):
    with open(path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index == row_index:
                return json.loads(line)
    raise IndexError(f"row_index {row_index} is outside {path}")


def prepare_sample(row, data_dir, sample_type):
    good_image_path = os.path.join(data_dir, row["good_image"])
    bad_image_path = os.path.join(data_dir, row["bad_image"])
    heatmap_path = os.path.join(data_dir, row["bad_heatmap"])

    if sample_type == "good":
        image_paths = [good_image_path]
        prompt = SINGLE_IMAGE_PROMPT
        reason = row["good_reason"]
        target = Image.new("RGB", Image.open(good_image_path).size)
    elif sample_type == "bad":
        image_paths = [bad_image_path]
        prompt = SINGLE_IMAGE_PROMPT
        reason = row["bad_reason"]
        target = pil_img2rgb(Image.open(heatmap_path))
    else:
        image_paths = [good_image_path, bad_image_path]
        prompt = PAIR_PROMPT
        reason = row["pair_reason"]
        target = pil_img2rgb(Image.open(heatmap_path))

    images = [pil_img2rgb(Image.open(path)) for path in image_paths]
    return images, image_paths, prompt, reason, target


def build_model_architecture(model_path):
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(os.path.join(model_path, "ae.safetensors"))
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config, meta=True
        )

    return model, vae_model


def build_model(model_path, checkpoint_path, device):
    model, vae_model = build_model_architecture(model_path)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map={"": device},
        dtype=torch.bfloat16,
    )

    lora_config = LoraConfig.from_pretrained(checkpoint_path)
    lora_config.inference_mode = True
    model = get_peft_model(model, lora_config)
    adapter_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
    adapter_state_dict = load_file(adapter_path, device="cpu")
    load_result = set_peft_model_state_dict(model, adapter_state_dict)
    print(load_result)
    if load_result.unexpected_keys:
        raise RuntimeError(f"Unexpected LoRA keys: {load_result.unexpected_keys}")
    del adapter_state_dict

    model.to(dtype=torch.bfloat16)
    model.requires_grad_(False)
    model.eval()
    vae_model.to(device).eval()
    return model, vae_model


@torch.inference_mode()
def generate_heatmap(
    inferencer,
    images,
    prompt,
    reason,
    cfg_text_scale,
    cfg_img_scale,
    num_timesteps,
    timestep_shift,
):
    outputs = inferencer.interleave_inference(
        [*images, prompt, f"<think>{reason}</think>"],
        think=False,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
    )
    return outputs[-1]


def run_inference(args, model_loader, metadata_extra=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference.")

    device = "cuda:0"
    torch.cuda.set_device(0)
    set_seed(args.seed)

    metadata_path = args.metadata_path or os.path.join(
        args.data_dir, "metadata", "val.jsonl"
    )
    row = load_jsonl_row(metadata_path, args.row_index)
    images, image_paths, prompt, reason, target = prepare_sample(
        row, args.data_dir, args.sample_type
    )

    model, vae_model = model_loader(args.model_path, args.checkpoint_path, device)
    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=DeviceImageTransform(
            ImageTransform(1024, 512, 16), device
        ),
        vit_transform=DeviceImageTransform(
            ImageTransform(518, 224, 14), device
        ),
        new_token_ids=new_token_ids,
    )

    default_device = torch.get_default_device()
    torch.set_default_device(device)
    try:
        prediction = generate_heatmap(
            inferencer=inferencer,
            images=images,
            prompt=prompt,
            reason=reason,
            cfg_text_scale=args.cfg_text_scale,
            cfg_img_scale=args.cfg_img_scale,
            num_timesteps=args.num_timesteps,
            timestep_shift=args.timestep_shift,
        )
    finally:
        torch.set_default_device(default_device)
    target = inferencer.vae_transform.resize_transform(target)

    checkpoint_name = os.path.basename(os.path.normpath(args.checkpoint_path))
    if checkpoint_name.endswith(".safetensors"):
        checkpoint_name = os.path.basename(
            os.path.dirname(os.path.normpath(args.checkpoint_path))
        )
    sample_dir = os.path.join(
        args.output_dir,
        f"{checkpoint_name}_row{args.row_index}_{args.sample_type}",
    )
    os.makedirs(sample_dir, exist_ok=True)
    prediction.save(os.path.join(sample_dir, "prediction.png"))
    target.save(os.path.join(sample_dir, "target.png"))
    for index, image in enumerate(images):
        image.save(os.path.join(sample_dir, f"input_{index}.png"))

    metadata = {
        "checkpoint_path": args.checkpoint_path,
        "metadata_path": metadata_path,
        "row_index": args.row_index,
        "sample_type": args.sample_type,
        "image_paths": image_paths,
        "prompt": prompt,
        "reason": reason,
        "seed": args.seed,
        "num_timesteps": args.num_timesteps,
        "timestep_shift": args.timestep_shift,
        "cfg_text_scale": args.cfg_text_scale,
        "cfg_img_scale": args.cfg_img_scale,
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    with open(os.path.join(sample_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved inference outputs to {sample_dir}")


def main():
    args = parse_args()
    run_inference(
        args,
        model_loader=build_model,
        metadata_extra={"lora_variant": args.lora_variant},
    )


if __name__ == "__main__":
    main()

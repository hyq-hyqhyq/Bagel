# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import random

import numpy as np
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from PIL import Image
from safetensors.torch import load_file

from data.data_utils import add_special_tokens, pil_img2rgb
from data.reason_heatmap_prompts import (
    PERSPECTIVE_REFINE_PROMPT,
    PERSPECTIVE_VERIFY_PROMPT,
    SANITY_REFINE_PROMPT,
    SANITY_VERIFY_PROMPT,
)
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
from sanity_patch.mask_utils import to_binary_mask
from sanity_patch.settings import (
    BINARY_MASK_THRESHOLD,
    DENSE_CFG_IMG_SCALE,
    DENSE_CFG_INTERVAL,
    DENSE_CFG_RENORM_MIN,
    DENSE_CFG_RENORM_TYPE,
    DENSE_CFG_TEXT_SCALE,
    SANITY_PATCH_PROMPT,
    TIMESTEP_SHIFT,
)


SAMPLE_TYPES = (
    "good",
    "bad",
    "pair",
    "good_refine",
    "bad_refine",
    "good_verify",
    "bad_verify",
)
PERSPECTIVE_SINGLE_PROMPT = (
    "Analyze the perspective and projection realism of this image."
)
PERSPECTIVE_PAIR_PROMPT = (
    "Compare the two images and explain the perspective and projection realism difference."
)
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
        description="Run single-GPU endpoint inference with a reason heatmap LoRA."
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
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument(
        "--sample_type",
        choices=SAMPLE_TYPES,
        default="bad",
    )
    parser.add_argument(
        "--prompt_domain",
        choices=("sanity", "perspective"),
        default="sanity",
    )
    parser.add_argument(
        "--two_round",
        action="store_true",
        help=(
            "Run refinement followed by verification, feeding the generated "
            "refined image into the second round."
        ),
    )
    parser.add_argument(
        "--two_round_quality",
        choices=("good", "bad"),
        default="bad",
        help="Choose the original image used by --two_round.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--timestep_shift", type=float, default=TIMESTEP_SHIFT)
    parser.add_argument("--cfg_text_scale", type=float, default=None)
    parser.add_argument("--cfg_img_scale", type=float, default=None)
    parser.add_argument(
        "--prompt_suffix",
        default="",
        help="Optional instruction appended to the dataset prompt.",
    )
    parser.add_argument(
        "--heatmap_only",
        action="store_true",
        help="Skip explanation generation and generate only the heatmap.",
    )
    parser.add_argument(
        "--binary_threshold",
        type=int,
        default=BINARY_MASK_THRESHOLD,
        help="Grayscale threshold used to save prediction.png as a binary mask.",
    )
    args = parser.parse_args()
    variant = LORA_VARIANTS[args.lora_variant]
    args.checkpoint_path = args.checkpoint_path or variant["checkpoint_path"]
    args.output_dir = args.output_dir or variant["output_dir"]
    if args.cfg_text_scale is None:
        args.cfg_text_scale = DENSE_CFG_TEXT_SCALE if args.heatmap_only else 4.0
    if args.cfg_img_scale is None:
        args.cfg_img_scale = DENSE_CFG_IMG_SCALE if args.heatmap_only else 2.0
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if not 0 <= args.binary_threshold <= 255:
        raise ValueError("binary_threshold must be between 0 and 255.")
    if args.two_round and args.heatmap_only:
        raise ValueError("--two_round cannot be combined with --heatmap_only.")
    if args.heatmap_only and args.sample_type not in ("good", "bad", "pair"):
        raise ValueError(
            "--heatmap_only only supports the legacy good, bad, and pair tasks."
        )
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


def get_domain_prompts(prompt_domain):
    if prompt_domain == "perspective":
        return PERSPECTIVE_REFINE_PROMPT, PERSPECTIVE_VERIFY_PROMPT
    return SANITY_REFINE_PROMPT, SANITY_VERIFY_PROMPT


def prepare_sample(row, data_dir, sample_type, prompt_domain="sanity"):
    good_image_path = os.path.join(data_dir, row["good_image"])
    bad_image_path = os.path.join(data_dir, row["bad_image"])
    heatmap_path = os.path.join(data_dir, row["bad_heatmap"])
    good_image = pil_img2rgb(Image.open(good_image_path))
    bad_image = pil_img2rgb(Image.open(bad_image_path))
    bad_heatmap = pil_img2rgb(Image.open(heatmap_path))
    black_heatmap = Image.new("RGB", good_image.size)

    refine_prompt, verify_prompt = get_domain_prompts(prompt_domain)
    if prompt_domain == "perspective":
        single_prompt = PERSPECTIVE_SINGLE_PROMPT
        pair_prompt = PERSPECTIVE_PAIR_PROMPT
    else:
        single_prompt = SANITY_PATCH_PROMPT
        pair_prompt = SANITY_PATCH_PROMPT

    if sample_type == "good":
        image_paths = [good_image_path]
        images = [good_image]
        prompt = single_prompt
        reason = row["good_reason"]
        target = black_heatmap
        output_type = "heatmap"
        target_score = row.get("good_score", 1.0)
    elif sample_type == "bad":
        image_paths = [bad_image_path]
        images = [bad_image]
        prompt = single_prompt
        reason = row["bad_reason"]
        target = bad_heatmap
        output_type = "heatmap"
        target_score = row.get("bad_score", 0.0)
    elif sample_type == "pair":
        image_paths = [good_image_path, bad_image_path]
        images = [good_image, bad_image]
        prompt = pair_prompt
        reason = row["pair_reason"]
        target = bad_heatmap
        output_type = "heatmap"
        target_score = row.get("pair_score")
    elif sample_type == "good_refine":
        image_paths = [good_image_path]
        images = [good_image]
        prompt = refine_prompt
        reason = row["good_reason"]
        target = good_image
        output_type = "image"
        target_score = row.get("good_score", 1.0)
    elif sample_type == "bad_refine":
        image_paths = [bad_image_path]
        images = [bad_image]
        prompt = refine_prompt
        reason = row["bad_reason"]
        target = good_image
        output_type = "image"
        target_score = row.get("bad_score", 0.0)
    elif sample_type == "good_verify":
        image_paths = [good_image_path, good_image_path]
        images = [good_image, good_image.copy()]
        prompt = verify_prompt
        reason = row["good_reason"]
        target = black_heatmap
        output_type = "heatmap"
        target_score = row.get("good_score", 1.0)
    elif sample_type == "bad_verify":
        image_paths = [bad_image_path, good_image_path]
        images = [bad_image, good_image]
        prompt = verify_prompt
        reason = row["bad_reason"]
        target = bad_heatmap
        output_type = "heatmap"
        target_score = row.get("bad_score", 0.0)
    else:
        raise ValueError(f"Unsupported sample_type: {sample_type}")

    return images, image_paths, prompt, reason, target, output_type, target_score


def build_model_architecture(
    model_path, score_head=False, split_gen_adapter_by_task=False
):
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.split_gen_expert_by_task = split_gen_adapter_by_task

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
        score_head=score_head,
        split_gen_adapter_by_task=split_gen_adapter_by_task,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config, meta=True
        )

    return model, vae_model


def normalize_lora_state_dict(state_dict):
    return {
        key.replace("_fsdp_wrapped_module.", "").replace(
            "_checkpoint_wrapped_module.", ""
        ): value
        for key, value in state_dict.items()
    }


def build_model(model_path, checkpoint_path, device):
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

    lora_config = LoraConfig.from_pretrained(checkpoint_path)
    modules_to_save = lora_config.modules_to_save or []
    score_head = "score_head" in modules_to_save
    model, vae_model = build_model_architecture(model_path)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map={"": device},
        dtype=torch.bfloat16,
    )

    if score_head:
        model.init_score_head()
        model.score_head.to(device=device, dtype=torch.bfloat16)
    lora_config.inference_mode = True
    model = get_peft_model(model, lora_config)
    adapter_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
    adapter_state_dict = normalize_lora_state_dict(
        load_file(adapter_path, device="cpu")
    )
    load_result = set_peft_model_state_dict(model, adapter_state_dict)
    missing_lora_keys = [
        key for key in load_result.missing_keys if ".lora_" in key
    ]
    if load_result.unexpected_keys or missing_lora_keys:
        raise RuntimeError(
            "LoRA adapter keys did not match the inference model: "
            f"missing_lora={missing_lora_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    print(
        "Loaded LoRA adapter; ignored "
        f"{len(load_result.missing_keys)} base-model keys."
    )
    del adapter_state_dict

    model.to(dtype=torch.bfloat16)
    model.requires_grad_(False)
    model.eval()
    vae_model.to(device).eval()
    return model, vae_model


@torch.inference_mode()
def generate_reason_heatmap(
    inferencer,
    images,
    prompt,
    cfg_text_scale,
    cfg_img_scale,
    num_timesteps,
    timestep_shift,
    gen_task,
    heatmap_only=False,
):
    dense_kwargs = {}
    if heatmap_only:
        dense_kwargs = {
            "cfg_interval": DENSE_CFG_INTERVAL,
            "cfg_renorm_min": DENSE_CFG_RENORM_MIN,
            "cfg_renorm_type": DENSE_CFG_RENORM_TYPE,
        }
    outputs = inferencer.interleave_inference(
        [*images, prompt],
        think=not heatmap_only,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        timestep_shift=timestep_shift,
        num_timesteps=num_timesteps,
        gen_task=gen_task,
        **dense_kwargs,
    )
    generated_reason = next(
        (item for item in outputs if isinstance(item, str)),
        None,
    )
    predicted_score = next(
        (item for item in outputs if isinstance(item, float)),
        None,
    )
    return generated_reason, predicted_score, outputs[-1]


def run_and_save_task(
    inferencer,
    args,
    images,
    image_paths,
    prompt,
    target_reason,
    target,
    output_type,
    target_score,
    sample_type,
    sample_dir,
    metadata_path,
    row_index,
    row,
    metadata_extra=None,
    task_metadata=None,
):
    prompt_suffix = args.prompt_suffix.strip()
    if prompt_suffix:
        prompt = f"{prompt} {prompt_suffix}"
    gen_task = "heatmap" if output_type == "heatmap" else "repair"
    generated_reason, predicted_score, prediction_raw = generate_reason_heatmap(
        inferencer=inferencer,
        images=images,
        prompt=prompt,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        num_timesteps=args.num_timesteps,
        timestep_shift=args.timestep_shift,
        gen_task=gen_task,
        heatmap_only=args.heatmap_only,
    )
    target_original = target.copy()
    target_model = inferencer.vae_transform.resize_transform(target)

    os.makedirs(sample_dir, exist_ok=True)
    prediction_raw.save(os.path.join(sample_dir, "prediction_raw.png"))
    prediction = prediction_raw.resize(
        images[0].size,
        resample=(
            Image.Resampling.NEAREST
            if output_type == "heatmap"
            else Image.Resampling.LANCZOS
        ),
    )
    if output_type == "heatmap":
        prediction = to_binary_mask(
            prediction,
            threshold=args.binary_threshold,
        )
    prediction.save(os.path.join(sample_dir, "prediction.png"))
    target_original.save(os.path.join(sample_dir, "target.png"))
    target_model.save(os.path.join(sample_dir, "target_model.png"))
    for index, image in enumerate(images):
        image.save(os.path.join(sample_dir, f"input_{index}.png"))
    if generated_reason is not None:
        with open(
            os.path.join(sample_dir, "reason.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(generated_reason)
    if predicted_score is not None:
        with open(
            os.path.join(sample_dir, "score.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"{predicted_score:.8f}\n")

    metadata = {
        "checkpoint_path": args.checkpoint_path,
        "metadata_path": metadata_path,
        "row_index": row_index,
        "source_row_index": row.get("source_row_index"),
        "sample_id": row.get("sample_id"),
        "source_split": row.get("split"),
        "sample_type": sample_type,
        "prompt_domain": args.prompt_domain,
        "output_type": output_type,
        "gen_task": gen_task,
        "predicted_score": predicted_score,
        "target_score": target_score,
        "image_paths": image_paths,
        "prompt": prompt,
        "heatmap_only": args.heatmap_only,
        "seed": args.seed,
        "num_timesteps": args.num_timesteps,
        "timestep_shift": args.timestep_shift,
        "cfg_text_scale": args.cfg_text_scale,
        "cfg_img_scale": args.cfg_img_scale,
        "binary_threshold": args.binary_threshold,
        "input_sizes": [list(image.size) for image in images],
        "target_original_size": list(target_original.size),
        "target_model_size": list(target_model.size),
        "prediction_size": list(prediction.size),
    }
    if args.heatmap_only:
        metadata.update(
            cfg_interval=list(DENSE_CFG_INTERVAL),
            cfg_renorm_min=DENSE_CFG_RENORM_MIN,
            cfg_renorm_type=DENSE_CFG_RENORM_TYPE,
        )
    else:
        metadata["target_reason"] = target_reason
    if metadata_extra:
        metadata.update(metadata_extra)
    if task_metadata:
        metadata.update(task_metadata)
    with open(
        os.path.join(sample_dir, "metadata.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved inference outputs to {sample_dir}")
    return prediction


def run_inference(args, model_loader, metadata_extra=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference.")

    device = "cuda:0"
    torch.cuda.set_device(0)
    set_seed(args.seed)

    metadata_path = args.metadata_path or os.path.join(
        args.data_dir, "metadata", "val.jsonl"
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

    checkpoint_name = os.path.basename(os.path.normpath(args.checkpoint_path))
    if checkpoint_name.endswith(".safetensors"):
        checkpoint_name = os.path.basename(
            os.path.dirname(os.path.normpath(args.checkpoint_path))
        )
    default_device = torch.get_default_device()
    torch.set_default_device(device)
    try:
        for row_index in range(args.row_index, args.row_index + args.num_samples):
            row = load_jsonl_row(metadata_path, row_index)
            if getattr(args, "two_round", False):
                quality = args.two_round_quality
                parent_dir = os.path.join(
                    args.output_dir,
                    f"{checkpoint_name}_row{row_index:04d}_{quality}_two_round",
                )
                round1_sample_type = f"{quality}_refine"
                round1 = prepare_sample(
                    row,
                    args.data_dir,
                    round1_sample_type,
                    args.prompt_domain,
                )
                round1_dir = os.path.join(parent_dir, "round1_refine")
                refined_image = run_and_save_task(
                    inferencer,
                    args,
                    *round1,
                    round1_sample_type,
                    round1_dir,
                    metadata_path,
                    row_index,
                    row,
                    metadata_extra,
                    {"two_round": True, "round": 1},
                )

                round2_sample_type = f"{quality}_verify"
                round2 = list(
                    prepare_sample(
                        row,
                        args.data_dir,
                        round2_sample_type,
                        args.prompt_domain,
                    )
                )
                round2[0][1] = refined_image
                round2_dir = os.path.join(parent_dir, "round2_verify")
                round1_prediction_path = os.path.join(
                    round1_dir, "prediction.png"
                )
                round2[1][1] = round1_prediction_path
                run_and_save_task(
                    inferencer,
                    args,
                    *round2,
                    round2_sample_type,
                    round2_dir,
                    metadata_path,
                    row_index,
                    row,
                    metadata_extra,
                    {
                        "two_round": True,
                        "round": 2,
                        "refined_image_path": round1_prediction_path,
                    },
                )
                continue

            (
                images,
                image_paths,
                prompt,
                target_reason,
                target,
                output_type,
                target_score,
            ) = prepare_sample(
                row,
                args.data_dir,
                args.sample_type,
                args.prompt_domain,
            )
            sample_dir = os.path.join(
                args.output_dir,
                f"{checkpoint_name}_row{row_index:04d}_{args.sample_type}",
            )
            run_and_save_task(
                inferencer,
                args,
                images,
                image_paths,
                prompt,
                target_reason,
                target,
                output_type,
                target_score,
                args.sample_type,
                sample_dir,
                metadata_path,
                row_index,
                row,
                metadata_extra,
                {"two_round": False},
            )
    finally:
        torch.set_default_device(default_device)


def main():
    args = parse_args()
    run_inference(
        args,
        model_loader=build_model,
        metadata_extra={"lora_variant": args.lora_variant},
    )


if __name__ == "__main__":
    main()

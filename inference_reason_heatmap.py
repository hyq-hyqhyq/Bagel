# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse

import torch
from accelerate import load_checkpoint_and_dispatch
from safetensors import safe_open

from inference_reason_heatmap_lora import (
    SAMPLE_TYPES,
    build_model_architecture,
    run_inference,
)
from sanity_patch.settings import (
    BINARY_MASK_THRESHOLD,
    DENSE_CFG_IMG_SCALE,
    DENSE_CFG_TEXT_SCALE,
    TIMESTEP_SHIFT,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run single-GPU teacher-forced inference with a full reason heatmap checkpoint."
    )
    parser.add_argument(
        "--model_path",
        default="/data/bagel/repo/agent/bpipe/models/BAGEL-7B-MoT",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=(
            "/data/bagel/repo/Bagel/results/reason_heatmap/checkpoints/"
            "0012000/model.safetensors"
        ),
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
        default="./results/reason_heatmap/inference",
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
    if args.cfg_text_scale is None:
        args.cfg_text_scale = DENSE_CFG_TEXT_SCALE if args.heatmap_only else 1.0
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


def build_model(model_path, checkpoint_path, device):
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        checkpoint_keys = tuple(f.keys())
        score_head = any(
            key.startswith("score_head.") or ".score_head." in key
            for key in checkpoint_keys
        )
        split_gen_adapter_by_task = any(
            "repair_gen_adapter." in key
            or "heatmap_gen_adapter." in key
            for key in checkpoint_keys
        )
    model, vae_model = build_model_architecture(
        model_path,
        score_head=score_head,
        split_gen_adapter_by_task=split_gen_adapter_by_task,
    )
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint_path,
        device_map={"": "cpu"},
        dtype=torch.bfloat16,
    )
    model.to(device=device, dtype=torch.bfloat16)
    model.requires_grad_(False)
    model.eval()
    vae_model.to(device).eval()
    return model, vae_model


def main():
    args = parse_args()
    run_inference(
        args,
        model_loader=build_model,
        metadata_extra={"checkpoint_type": "full_model"},
    )


if __name__ == "__main__":
    main()

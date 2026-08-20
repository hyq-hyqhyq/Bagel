# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse

import torch
from accelerate import load_checkpoint_and_dispatch

from inference_reason_heatmap_lora import build_model_architecture, run_inference
from sanity_patch.settings import BINARY_MASK_THRESHOLD, TIMESTEP_SHIFT


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
        choices=("good", "bad", "pair"),
        default="bad",
    )
    parser.add_argument(
        "--output_dir",
        default="./results/reason_heatmap/inference",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--timestep_shift", type=float, default=TIMESTEP_SHIFT)
    parser.add_argument("--cfg_text_scale", type=float, default=1.0)
    parser.add_argument("--cfg_img_scale", type=float, default=2.0)
    parser.add_argument(
        "--prompt_suffix",
        default="",
        help="Optional instruction appended to the dataset prompt.",
    )
    parser.add_argument(
        "--binary_threshold",
        type=int,
        default=BINARY_MASK_THRESHOLD,
        help="Grayscale threshold used to save prediction.png as a binary mask.",
    )
    args = parser.parse_args()
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if not 0 <= args.binary_threshold <= 255:
        raise ValueError("binary_threshold must be between 0 and 255.")
    return args


def build_model(model_path, checkpoint_path, device):
    model, vae_model = build_model_architecture(model_path)
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

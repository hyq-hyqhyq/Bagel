# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""On-policy explanation rollout followed by token-only judgment CE.

The explanation is generated once, without an autograd graph, and then reused
for every local check and the global verdict.  A second, parallel text forward
inserts the gold verdict tokens after the generated check blocks and computes
CE only at those verdict positions.  This keeps the expensive autoregressive
part to one rollout per logical sample.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from data.perspective_judgment_utils import (
    build_judgment_sequence,
    count_generated_checks,
)

from .e2e import (
    _generate_reason_ids,
    _new_context,
    _update_image,
    _update_text,
)


def forward_judgment_rollout(
    model,
    inputs: Dict[str, Any],
    vae_model,
    options: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """Run cache-based generation without changing the outer train mode."""
    was_training = model.training
    model.eval()
    try:
        return _forward_judgment_rollout_impl(
            model, inputs, vae_model=vae_model, options=options
        )
    finally:
        model.train(was_training)


def _forward_judgment_rollout_impl(
    model,
    inputs: Dict[str, Any],
    vae_model,
    options: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """Generate one explanation and supervise all verdict labels."""
    if vae_model is None:
        raise ValueError("Judgment rollout requires a VAE model")
    tokenizer = options.get("tokenizer")
    special_tokens = options.get("special_tokens")
    if tokenizer is None or special_tokens is None:
        raise ValueError("Judgment rollout requires tokenizer and special tokens")

    vae_images = inputs.get("vae_images") or []
    vit_images = inputs.get("vit_images") or []
    if len(vae_images) != len(vit_images) or not vae_images:
        raise ValueError("Judgment rollout requires aligned VAE/VIT images")
    device = next(model.parameters()).device
    max_tokens = int(options.get("max_tokens", 320))
    if max_tokens < 1:
        raise ValueError("judgment rollout max_tokens must be positive")

    # The discrete rollout and its multimodal prefix deliberately have no
    # graph.  Judgment gradients begin at the second text forward below.
    with torch.no_grad():
        context = _new_context(model)
        for vae_image, vit_image in zip(vae_images, vit_images):
            vae_image = vae_image.to(device=device, non_blocking=True)
            vit_image = vit_image.to(device=device, non_blocking=True)
            latent = vae_model.encode(vae_image.unsqueeze(0))
            _update_image(
                model,
                context,
                latent,
                vit_image,
                special_tokens,
                inputs.get("gen_task", "heatmap"),
            )
        _update_text(model, context, inputs["prompt_ids"], special_tokens)
        generated_ids = _generate_reason_ids(
            model,
            context,
            special_tokens,
            tokenizer,
            max_tokens,
        )
        sequence = build_judgment_sequence(
            tokenizer,
            generated_ids.tolist(),
            [int(value) for value in inputs["check_labels"]],
            int(inputs["global_label"]),
        )

    sequence_ids, label_positions, label_ids, label_kinds = sequence
    hidden = _update_text(
        model,
        context,
        sequence_ids,
        special_tokens,
        return_hidden=True,
    )
    positions = torch.tensor(label_positions, device=device, dtype=torch.long)
    targets = torch.tensor(label_ids, device=device, dtype=torch.long)
    kinds = torch.tensor(label_kinds, device=device, dtype=torch.long)
    logits = model.language_model.lm_head(hidden[positions])
    per_token = F.cross_entropy(logits.float(), targets, reduction="none")

    check_mask = kinds == 1
    global_mask = kinds == 2
    if not bool(check_mask.any()) or not bool(global_mask.any()):
        raise RuntimeError("Judgment rollout produced incomplete label masks")
    return {
        "check_ce": per_token[check_mask].mean(),
        "global_ce": per_token[global_mask].mean(),
        "generated_tokens": per_token.new_tensor(float(generated_ids.numel())),
        "structure_fallback": per_token.new_tensor(
            float(
                count_generated_checks(tokenizer, generated_ids.tolist())
                != len(inputs["check_labels"])
            )
        ),
    }

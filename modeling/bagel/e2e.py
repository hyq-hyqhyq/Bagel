# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Differentiable two-stage objective for perspective repair and heatmaps.

This module is imported only when ``Bagel.forward`` receives ``e2e_inputs``.
Legacy training and inference therefore keep their original code path.
"""

import copy
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from data.data_utils import patchify

from .qwen2_navit import NaiveCache


def pack_latent(model, latent):
    """Convert BCHW VAE latent into BAGEL's packed latent-token layout."""
    if latent.ndim != 4 or latent.shape[0] != 1:
        raise ValueError("E2E currently requires one BCHW latent per rank")
    p = model.latent_patch_size
    _, channels, height, width = latent.shape
    if channels != model.latent_channel or height % p or width % p:
        raise ValueError(
            f"Invalid latent shape {tuple(latent.shape)} for patch size {p}"
        )
    h, w = height // p, width // p
    packed = latent[0].reshape(channels, h, p, w, p)
    packed = torch.einsum("chpwq->hwpqc", packed)
    return packed.reshape(h * w, p * p * channels), (h, w)


def unpack_latent(model, packed, image_size):
    """Convert packed latent tokens back to BCHW without detaching."""
    height, width = image_size
    h = height // model.latent_downsample
    w = width // model.latent_downsample
    p = model.latent_patch_size
    latent = packed.reshape(1, h, w, p, p, model.latent_channel)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    return latent.reshape(1, model.latent_channel, h * p, w * p)


def _new_context(model):
    return {
        "cache": NaiveCache(model.config.llm_config.num_hidden_layers),
        "kv_len": 0,
        "rope": 0,
        "vae_hidden_sum": None,
        "vae_count": 0,
        "vit_hidden_sum": None,
        "vit_count": 0,
        "text_hidden": None,
    }


def _copy_context(context):
    return copy.deepcopy(context)


def _as_ids(value, device):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.long).flatten()
    return torch.tensor(value, device=device, dtype=torch.long).flatten()


def _update_text(model, context, token_ids, special_tokens, return_hidden=False):
    device = next(model.parameters()).device
    token_ids = _as_ids(token_ids, device)
    bos = int(special_tokens["bos_token_id"])
    eos = int(special_tokens["eos_token_id"])
    query_ids = torch.cat(
        [
            token_ids.new_tensor([bos]),
            token_ids,
            token_ids.new_tensor([eos]),
        ]
    )
    query_len = query_ids.numel()
    kv_len = context["kv_len"]
    rope = context["rope"]
    kwargs = {
        "packed_text_ids": query_ids,
        "packed_text_position_ids": torch.arange(
            rope, rope + query_len, device=device, dtype=torch.long
        ),
        "text_token_lens": torch.tensor(
            [query_len], device=device, dtype=torch.int
        ),
        "packed_text_indexes": torch.arange(
            kv_len, kv_len + query_len, device=device, dtype=torch.long
        ),
        "packed_key_value_indexes": torch.arange(
            kv_len, device=device, dtype=torch.long
        ),
        "key_values_lens": torch.tensor(
            [kv_len], device=device, dtype=torch.int
        ),
    }
    result = model.forward_cache_update_text(
        context["cache"], return_hidden=return_hidden, **kwargs
    )
    if return_hidden:
        context["cache"], hidden = result
        context["text_hidden"] = hidden[-1]
    else:
        context["cache"] = result
        hidden = None
    context["kv_len"] += query_len
    context["rope"] += query_len
    return hidden


def _update_vae(model, context, latent, special_tokens, gen_task):
    device = latent.device
    packed, (h, w) = pack_latent(model, latent)
    num_tokens = packed.shape[0]
    kv_len = context["kv_len"]
    rope = context["rope"]
    query_len = num_tokens + 2
    image_height = h * model.latent_downsample
    image_width = w * model.latent_downsample
    text_indexes = torch.tensor(
        [0, num_tokens + 1], device=device, dtype=torch.long
    )
    vae_indexes = torch.arange(1, num_tokens + 1, device=device)
    kwargs = {
        "vae_model": None,
        "past_key_values": context["cache"],
        "padded_images": None,
        "padded_latent": latent,
        "patchified_vae_latent_shapes": [(h, w)],
        "packed_vae_position_ids": model.get_flattened_position_ids(
            image_height,
            image_width,
            model.latent_downsample,
            max_num_patches_per_side=model.max_latent_size,
        ).to(device),
        "packed_timesteps": torch.zeros(1, device=device),
        "packed_vae_token_indexes": vae_indexes,
        "packed_text_ids": torch.tensor(
            [
                int(special_tokens["start_of_image"]),
                int(special_tokens["end_of_image"]),
            ],
            device=device,
            dtype=torch.long,
        ),
        "packed_text_indexes": text_indexes,
        "packed_position_ids": torch.full(
            (query_len,), rope, device=device, dtype=torch.long
        ),
        "packed_seqlens": torch.tensor(
            [query_len], device=device, dtype=torch.int
        ),
        "packed_indexes": torch.arange(
            kv_len, kv_len + query_len, device=device, dtype=torch.long
        ),
        "key_values_lens": torch.tensor(
            [kv_len], device=device, dtype=torch.int
        ),
        "packed_key_value_indexes": torch.arange(
            kv_len, device=device, dtype=torch.long
        ),
        "return_hidden": True,
        "gen_task": gen_task,
    }
    context["cache"], hidden = model.forward_cache_update_vae(**kwargs)
    image_hidden = hidden[vae_indexes]
    image_sum = image_hidden.sum(dim=0)
    if context["vae_hidden_sum"] is None:
        context["vae_hidden_sum"] = image_sum
    else:
        context["vae_hidden_sum"] = context["vae_hidden_sum"] + image_sum
    context["vae_count"] += num_tokens
    context["kv_len"] += query_len
    context["rope"] += 1


def _update_vit(model, context, image, special_tokens):
    device = image.device
    vit_tokens = patchify(image, model.vit_patch_size)
    num_tokens = vit_tokens.shape[0]
    kv_len = context["kv_len"]
    rope = context["rope"]
    query_len = num_tokens + 2
    text_indexes = torch.tensor(
        [0, num_tokens + 1], device=device, dtype=torch.long
    )
    vit_indexes = torch.arange(1, num_tokens + 1, device=device)
    context["cache"], hidden = model.forward_cache_update_vit(
        past_key_values=context["cache"],
        packed_text_ids=torch.tensor(
            [
                int(special_tokens["start_of_image"]),
                int(special_tokens["end_of_image"]),
            ],
            device=device,
            dtype=torch.long,
        ),
        packed_text_indexes=text_indexes,
        packed_vit_tokens=vit_tokens,
        packed_vit_token_indexes=vit_indexes,
        packed_vit_position_ids=model.get_flattened_position_ids(
            image.shape[-2],
            image.shape[-1],
            model.vit_patch_size,
            max_num_patches_per_side=model.vit_max_num_patch_per_side,
        ).to(device),
        vit_token_seqlens=torch.tensor(
            [num_tokens], device=device, dtype=torch.int
        ),
        packed_position_ids=torch.full(
            (query_len,), rope, device=device, dtype=torch.long
        ),
        packed_seqlens=torch.tensor(
            [query_len], device=device, dtype=torch.int
        ),
        packed_indexes=torch.arange(
            kv_len, kv_len + query_len, device=device, dtype=torch.long
        ),
        packed_key_value_indexes=torch.arange(
            kv_len, device=device, dtype=torch.long
        ),
        key_values_lens=torch.tensor(
            [kv_len], device=device, dtype=torch.int
        ),
        return_hidden=True,
    )
    image_hidden = hidden[vit_indexes]
    image_sum = image_hidden.sum(dim=0)
    if context["vit_hidden_sum"] is None:
        context["vit_hidden_sum"] = image_sum
    else:
        context["vit_hidden_sum"] = context["vit_hidden_sum"] + image_sum
    context["vit_count"] += num_tokens
    context["kv_len"] += query_len
    context["rope"] += 1


def _update_image(model, context, latent, vit_image, special_tokens, gen_task):
    _update_vae(model, context, latent, special_tokens, gen_task)
    _update_vit(model, context, vit_image, special_tokens)


def _reason_ce(model, context, reason_ids, special_tokens):
    hidden = _update_text(
        model, context, reason_ids, special_tokens, return_hidden=True
    )
    reason_ids = _as_ids(reason_ids, hidden.device)
    labels = torch.cat(
        [
            reason_ids,
            reason_ids.new_tensor([int(special_tokens["eos_token_id"])]),
        ]
    )
    logits = model.language_model.lm_head(hidden[:-1])
    return F.cross_entropy(logits.float(), labels), context


def _score_loss(model, context, score_label):
    if not hasattr(model, "score_head"):
        raise ValueError("E2E training requires --score_head True")
    if (
        context["text_hidden"] is None
        or context["vae_hidden_sum"] is None
        or context["vit_hidden_sum"] is None
    ):
        raise RuntimeError("Score context is missing text, VAE, or ViT features")
    hidden = torch.cat(
        [
            context["text_hidden"],
            context["vae_hidden_sum"] / max(context["vae_count"], 1),
            context["vit_hidden_sum"] / max(context["vit_count"], 1),
        ],
        dim=-1,
    )
    pred = model.score_head(hidden).float().squeeze()
    label = pred.new_tensor(float(score_label))
    return (pred - label).square(), pred


def _generation_query(model, context, image_size, special_tokens, noise=None):
    device = next(model.parameters()).device
    height, width = image_size
    h = height // model.latent_downsample
    w = width // model.latent_downsample
    num_tokens = h * w
    query_len = num_tokens + 2
    kv_len = context["kv_len"]
    if noise is None:
        noise = torch.randn(
            num_tokens,
            model.patch_latent_dim,
            device=device,
            dtype=next(model.parameters()).dtype,
        )
    return {
        "packed_text_ids": torch.tensor(
            [
                int(special_tokens["start_of_image"]),
                int(special_tokens["end_of_image"]),
            ],
            device=device,
            dtype=torch.long,
        ),
        "packed_text_indexes": torch.tensor(
            [0, num_tokens + 1], device=device, dtype=torch.long
        ),
        "packed_init_noises": noise,
        "packed_vae_position_ids": model.get_flattened_position_ids(
            height,
            width,
            model.latent_downsample,
            max_num_patches_per_side=model.max_latent_size,
        ).to(device),
        "packed_vae_token_indexes": torch.arange(
            1, num_tokens + 1, device=device, dtype=torch.long
        ),
        "packed_seqlens": torch.tensor(
            [query_len], device=device, dtype=torch.int
        ),
        "packed_position_ids": torch.full(
            (query_len,), context["rope"], device=device, dtype=torch.long
        ),
        "packed_indexes": torch.arange(
            kv_len, kv_len + query_len, device=device, dtype=torch.long
        ),
        "past_key_values": context["cache"],
        "key_values_lens": torch.tensor(
            [kv_len], device=device, dtype=torch.int
        ),
        "packed_key_value_indexes": torch.arange(
            kv_len, device=device, dtype=torch.long
        ),
    }


def _cfg_query(model, context, image_size):
    device = next(model.parameters()).device
    height, width = image_size
    num_tokens = (
        height // model.latent_downsample
    ) * (width // model.latent_downsample)
    query_len = num_tokens + 2
    kv_len = context["kv_len"]
    return {
        "packed_position_ids": torch.full(
            (query_len,), context["rope"], device=device, dtype=torch.long
        ),
        "packed_query_indexes": torch.arange(
            kv_len, kv_len + query_len, device=device, dtype=torch.long
        ),
        "key_values_lens": torch.tensor(
            [kv_len], device=device, dtype=torch.int
        ),
        "past_key_values": context["cache"],
        "packed_key_value_indexes": torch.arange(
            kv_len, device=device, dtype=torch.long
        ),
    }


def truncated_sample(
    model,
    context,
    cfg_text_context,
    image_size,
    special_tokens,
    num_timesteps,
    gradient_steps,
    timestep_shift,
    cfg_text_scale,
):
    """Use the legacy schedule, retaining autograd for only its last K updates."""
    query = _generation_query(model, context, image_size, special_tokens)
    cfg = _cfg_query(model, cfg_text_context, image_size)
    x_t = query.pop("packed_init_noises")
    timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
    timesteps = timestep_shift * timesteps / (
        1 + (timestep_shift - 1) * timesteps
    )
    dts = timesteps[:-1] - timesteps[1:]
    timesteps = timesteps[:-1]
    if not 0 < gradient_steps <= len(timesteps):
        raise ValueError(
            f"gradient_steps must be in [1, {len(timesteps)}]"
        )
    grad_start = len(timesteps) - gradient_steps

    for index, timestep_value in enumerate(timesteps):
        grad_enabled = index >= grad_start
        with torch.set_grad_enabled(grad_enabled):
            # Keep timesteps in float32.  BAGEL's flow path validates that all
            # packed tokens share one timestep with Tensor.unique(), which is
            # not implemented for bfloat16 on the CUDA/PyTorch versions used
            # by the training environment.
            timestep = timestep_value.float().expand(x_t.shape[0])
            velocity = model._forward_flow(
                x_t=x_t,
                timestep=timestep,
                gen_task="repair",
                route_gen_task=True,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=1.0,
                cfg_text_packed_position_ids=cfg["packed_position_ids"],
                cfg_text_packed_query_indexes=cfg["packed_query_indexes"],
                cfg_text_key_values_lens=cfg["key_values_lens"],
                cfg_text_past_key_values=cfg["past_key_values"],
                cfg_text_packed_key_value_indexes=cfg[
                    "packed_key_value_indexes"
                ],
                **query,
            )
            x_t = x_t - velocity.to(x_t.device) * dts[index]
    return x_t


def _flow_loss(model, context, target_latent, image_size, special_tokens, task):
    clean, _ = pack_latent(model, target_latent)
    noise = torch.randn_like(clean)
    logit_t = torch.randn((), device=clean.device, dtype=torch.float32)
    timestep = torch.sigmoid(logit_t)
    timestep = model.timestep_shift * timestep / (
        1 + (model.timestep_shift - 1) * timestep
    )
    x_t = (1 - timestep) * clean + timestep * noise
    query = _generation_query(
        model, context, image_size, special_tokens, noise=x_t
    )
    query.pop("packed_init_noises")
    velocity = model._forward_flow(
        x_t=x_t,
        timestep=timestep.float().expand(x_t.shape[0]),
        gen_task=task,
        route_gen_task=True,
        **query,
    )
    target = noise - clean
    return (velocity.float() - target.float()).square().mean()


@torch.no_grad()
def _generate_reason_ids(
    model, context, special_tokens, tokenizer, max_reason_tokens
):
    device = next(model.parameters()).device
    start = model.prepare_start_tokens(
        [context["kv_len"]], [context["rope"]], special_tokens
    )
    start = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in start.items()
    }
    generated = model.generate_text(
        past_key_values=_copy_context(context)["cache"],
        max_length=max_reason_tokens,
        do_sample=False,
        temperature=1.0,
        end_token_id=int(special_tokens["eos_token_id"]),
        **start,
    )
    decoded = tokenizer.decode(generated[:, 0])
    decoded = decoded.split("<|im_end|>")[0]
    if "<|im_start|>" in decoded:
        decoded = decoded.split("<|im_start|>", 1)[1]
    return torch.tensor(
        tokenizer.encode(decoded), device=device, dtype=torch.long
    )


def _tensor_value(inputs, key, device):
    value = inputs[key]
    if not torch.is_tensor(value):
        value = torch.tensor(value)
    return value.to(device)


def forward_e2e(model, inputs: Dict[str, Any], vae_model, options):
    """Compute one of the three independent E2E loss phases.

    The two teacher-forced Task-1 objectives use separate forward graphs:
    ``repair_reason_score`` supervises reasoning and score, and
    ``repair_flow`` supervises flow matching. ``heatmap`` then builds the
    sampled repair-to-heatmap graph. A combined repair phase is deliberately
    unsupported so no two phase graphs can coexist before backward.
    """
    if vae_model is None:
        raise ValueError("e2e_vae_model is required")
    tokenizer = options.get("tokenizer")
    special_tokens = options.get("special_tokens")
    if tokenizer is None or special_tokens is None:
        raise ValueError("E2E options require tokenizer and special_tokens")
    phase = options.get("phase")
    if phase not in ("repair_reason_score", "repair_flow", "heatmap"):
        raise ValueError(
            "E2E phase must be one of: repair_reason_score, repair_flow, "
            "heatmap; "
            f"got {phase!r}"
        )

    device = next(model.parameters()).device
    original_latent = _tensor_value(inputs, "original_latent", device)
    refined_latent = _tensor_value(inputs, "refined_latent", device)
    heatmap_latent = _tensor_value(inputs, "heatmap_latent", device)
    original_vit = _tensor_value(inputs, "original_vit_image", device)
    if original_vit.ndim == 4:
        original_vit = original_vit[0]
    image_size_value = inputs["image_size"]
    if torch.is_tensor(image_size_value):
        image_size_value = image_size_value.flatten().tolist()
    image_size = tuple(int(item) for item in image_size_value)
    score_label = inputs["score_label"]
    if torch.is_tensor(score_label):
        score_label = score_label.item()

    system_ids = inputs["system_ids"]
    repair_prompt_ids = inputs["repair_prompt_ids"]
    heatmap_prompt_ids = inputs["heatmap_prompt_ids"]
    repair_reason_ids = inputs["repair_reason_ids"]
    heatmap_reason_ids = inputs["heatmap_reason_ids"]

    losses = {}
    if phase in ("repair_reason_score", "repair_flow"):
        # Both repair phases deliberately rebuild the common prefix. This
        # costs one extra forward, but lets each graph be backwarded and
        # released before the next phase is constructed.
        repair_context = _new_context(model)
        _update_text(model, repair_context, system_ids, special_tokens)
        _update_image(
            model,
            repair_context,
            original_latent,
            original_vit,
            special_tokens,
            "repair",
        )
        _update_text(model, repair_context, repair_prompt_ids, special_tokens)

        if phase == "repair_reason_score":
            repair_ce, repair_context = _reason_ce(
                model, repair_context, repair_reason_ids, special_tokens
            )
            repair_score, repair_score_pred = _score_loss(
                model, repair_context, score_label
            )
            losses.update(
                repair_reason=repair_ce,
                repair_score=repair_score,
                repair_score_pred=repair_score_pred.detach(),
            )
            return losses

        # Recreate the exact teacher-forced reason context needed by the
        # image-flow objective, without retaining LM logits for a CE loss.
        _update_text(
            model,
            repair_context,
            repair_reason_ids,
            special_tokens,
            return_hidden=False,
        )
        repair_flow = _flow_loss(
            model,
            repair_context,
            refined_latent,
            image_size,
            special_tokens,
            "repair",
        )
        losses.update(repair_flow=repair_flow)
        return losses

    # Inference-faithful Task-1 prefill and autoregressive reason are discrete
    # and intentionally graph-free.  Only the final K Euler updates retain a
    # graph, which is the truncated-backprop boundary requested for this run.
    with torch.no_grad():
        sample_context = _new_context(model)
        _update_text(model, sample_context, system_ids, special_tokens)
        _update_image(
            model,
            sample_context,
            original_latent,
            original_vit,
            special_tokens,
            "repair",
        )
        cfg_text_context = _copy_context(sample_context)
        _update_text(
            model, sample_context, repair_prompt_ids, special_tokens
        )
        generated_reason_ids = _generate_reason_ids(
            model,
            sample_context,
            special_tokens,
            tokenizer,
            int(options.get("max_reason_tokens", 1000)),
        )
        _update_text(
            model, sample_context, generated_reason_ids, special_tokens
        )

    predicted_packed = truncated_sample(
        model,
        sample_context,
        cfg_text_context,
        image_size,
        special_tokens,
        num_timesteps=int(options.get("num_timesteps", 50)),
        gradient_steps=int(options.get("gradient_steps", 4)),
        timestep_shift=float(options.get("timestep_shift", 4.0)),
        cfg_text_scale=float(options.get("cfg_text_scale", 4.0)),
    )
    predicted_latent = unpack_latent(model, predicted_packed, image_size)

    # Latent + ViT handoff: no PIL conversion and no VAE re-encode. Decoder
    # weights remain frozen, but autograd traverses the decoder into Task 1.
    # Non-reentrant checkpointing discards the decoder's intermediate
    # activations and recomputes them during backward without detaching the
    # heatmap-loss gradient from the predicted repair latent.
    if bool(options.get("vae_decoder_checkpoint", False)):
        predicted_image = activation_checkpoint(
            vae_model.decode,
            predicted_latent,
            use_reentrant=False,
        )
    else:
        predicted_image = vae_model.decode(predicted_latent)
    predicted_image = predicted_image.clamp(-1, 1)
    predicted_vit = F.interpolate(
        predicted_image.float(),
        size=tuple(original_vit.shape[-2:]),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).to(original_vit.dtype)[0]

    # Task-2 supervision is conditioned on the actual Task-1 prediction.  All
    # three Task-2 losses therefore propagate through latent and ViT handoff
    # and through the final K Task-1 denoising updates.
    heatmap_context = _new_context(model)
    _update_text(model, heatmap_context, system_ids, special_tokens)
    _update_image(
        model,
        heatmap_context,
        original_latent,
        original_vit,
        special_tokens,
        "heatmap",
    )
    _update_image(
        model,
        heatmap_context,
        predicted_latent,
        predicted_vit,
        special_tokens,
        "heatmap",
    )
    _update_text(
        model, heatmap_context, heatmap_prompt_ids, special_tokens
    )
    heatmap_ce, heatmap_context = _reason_ce(
        model, heatmap_context, heatmap_reason_ids, special_tokens
    )
    heatmap_score, heatmap_score_pred = _score_loss(
        model, heatmap_context, score_label
    )
    heatmap_flow = _flow_loss(
        model,
        heatmap_context,
        heatmap_latent,
        image_size,
        special_tokens,
        "heatmap",
    )

    losses.update(
        heatmap_flow=heatmap_flow,
        heatmap_reason=heatmap_ce,
        heatmap_score=heatmap_score,
        heatmap_score_pred=heatmap_score_pred.detach(),
        generated_reason_tokens=heatmap_flow.new_tensor(
            generated_reason_ids.numel()
        ),
    )
    return losses

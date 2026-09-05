# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Data and loop helpers for isolated truncated two-stage fine-tuning."""

import gc
import os
from contextlib import nullcontext
from time import time

import torch
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader

from data.interleave_datasets.perspective_e2e_dataset import (
    PerspectiveE2EIterableDataset,
)
from data.reason_heatmap_dataset_info import DATASET_INFO
from data.transforms import ImageTransform
from train.fsdp_utils import FSDPCheckpoint, fsdp_ema_update


LOSS_WEIGHTS = {
    "heatmap_flow": 10.0,
    "repair_flow": 1.0,
    "heatmap_score": 1.0,
    "repair_score": 0.2,
    "heatmap_reason": 0.25,
    "repair_reason": 0.05,
}

REPAIR_REASON_SCORE_LOSS_NAMES = ("repair_score", "repair_reason")
REPAIR_FLOW_LOSS_NAMES = ("repair_flow",)
HEATMAP_LOSS_NAMES = ("heatmap_flow", "heatmap_score", "heatmap_reason")


def _single_item_collate(batch):
    if len(batch) != 1:
        raise ValueError("E2E training requires batch_size=1 per rank")
    return batch[0]


def build_e2e_train_loader(
    dataset_meta,
    tokenizer,
    local_rank,
    world_size,
    num_workers,
    prefetch_factor,
    data_seed,
    data_status,
):
    if list(dataset_meta) != ["perspective_e2e"]:
        raise ValueError(
            "The E2E entrypoint requires one perspective_e2e dataset group"
        )
    args = dict(dataset_meta["perspective_e2e"])
    dataset_names = args.pop("dataset_names")
    if dataset_names != ["perspective"]:
        raise ValueError("perspective_e2e requires dataset_names: [perspective]")
    args.pop("is_mandatory", None)
    args.pop("weight", None)
    transform = ImageTransform(**args.pop("image_transform_args"))
    vit_transform = ImageTransform(**args.pop("vit_image_transform_args"))
    entry = DATASET_INFO["perspective_e2e"]["perspective"]
    if not entry["data_dir"] or not entry["jsonl_path"]:
        raise ValueError(
            "Set BAGEL_REASON_HEATMAP_DATA_DIR and "
            "BAGEL_REASON_HEATMAP_METADATA_PATH before E2E training"
        )
    resume_status = None
    if data_status and "perspective_e2e" in data_status:
        resume_status = data_status["perspective_e2e"]
    dataset = PerspectiveE2EIterableDataset(
        dataset_name="perspective_e2e",
        transform=transform,
        tokenizer=tokenizer,
        vit_transform=vit_transform,
        jsonl_path_list=[entry["jsonl_path"]],
        data_dir_list=[entry["data_dir"]],
        num_used_data=args.pop("num_used_data"),
        local_rank=local_rank,
        world_size=world_size,
        num_workers=num_workers,
        data_status=resume_status,
        **args,
    )
    if args:
        raise ValueError(f"Unknown perspective_e2e config keys: {sorted(args)}")
    dataset.set_epoch(data_seed)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": 1,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": _single_item_collate,
        "drop_last": True,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    return value


def _release_phase_cache(enabled):
    """Release only unreachable/cached allocations between E2E phases."""
    if not enabled:
        return
    gc.collect()
    torch.cuda.empty_cache()


def _stage_parameter_gradients(model, staged_gradients):
    """Accumulate this rank's reduced FSDP gradients in CPU memory."""
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        cpu_gradient = gradient.detach().to(
            device="cpu", non_blocking=False, copy=True
        )
        parameter_id = id(parameter)
        if parameter_id in staged_gradients:
            staged_gradients[parameter_id][1].add_(cpu_gradient)
        else:
            staged_gradients[parameter_id] = [parameter, cpu_gradient]
        parameter.grad = None


def _restore_parameter_gradients(staged_gradients):
    """Add CPU-staged repair gradients to the live heatmap gradients."""
    for parameter, cpu_gradient in staged_gradients.values():
        target_dtype = (
            parameter.grad.dtype
            if parameter.grad is not None
            else parameter.dtype
        )
        restored_gradient = cpu_gradient.to(
            device=parameter.device,
            dtype=target_dtype,
            non_blocking=False,
        )
        if parameter.grad is None:
            parameter.grad = restored_gradient
        else:
            parameter.grad.add_(restored_gradient)
        del restored_gradient
    staged_gradients.clear()


def _prepare_e2e_inputs(data, vae_model, device):
    data = dict(data)
    data_indexes = data.pop("data_indexes", None)
    original_image = _to_device(data.pop("original_vae_image"), device)
    refined_image = _to_device(data.pop("refined_vae_image"), device)
    heatmap_image = _to_device(data.pop("heatmap_vae_image"), device)
    with torch.no_grad(), torch.amp.autocast(
        "cuda", enabled=True, dtype=torch.bfloat16
    ):
        data["original_latent"] = vae_model.encode(original_image.unsqueeze(0))
        data["refined_latent"] = vae_model.encode(refined_image.unsqueeze(0))
        data["heatmap_latent"] = vae_model.encode(heatmap_image.unsqueeze(0))
    for key, value in tuple(data.items()):
        data[key] = _to_device(value, device)
    return data, data_indexes


def _gather_data_status(data_status):
    if dist.get_rank() == 0:
        gathered = [None] * dist.get_world_size()
    else:
        gathered = None
    dist.gather_object(data_status, gathered, dst=0)
    return gathered


def _save_checkpoint(
    step,
    fsdp_model,
    ema_model,
    optimizer,
    scheduler,
    logger,
    fsdp_config,
    checkpoint_dir,
    data_status,
):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gathered = _gather_data_status(data_status)
    FSDPCheckpoint.fsdp_save_ckpt(
        ckpt_dir=checkpoint_dir,
        train_steps=step,
        model=fsdp_model,
        ema_model=ema_model,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        fsdp_config=fsdp_config,
        data_status=gathered,
    )
    gc.collect()
    torch.cuda.empty_cache()


def run_e2e_training(
    fsdp_model,
    vae_model,
    tokenizer,
    special_tokens,
    train_loader,
    optimizer,
    scheduler,
    train_step,
    data_status,
    training_args,
    fsdp_config,
    logger,
    ema_model,
    device,
):
    if training_args.ema_enabled and ema_model is None:
        raise RuntimeError("ema_enabled=True but no EMA model was constructed")
    if training_args.cfg_img_scale != 1.0:
        raise ValueError(
            "The first E2E implementation requires cfg_img_scale=1.0"
        )
    options = {
        "tokenizer": tokenizer,
        "special_tokens": special_tokens,
        "num_timesteps": training_args.num_timesteps,
        "gradient_steps": training_args.gradient_denoise_steps,
        "max_reason_tokens": training_args.max_reason_tokens,
        "vae_decoder_checkpoint": training_args.vae_decoder_checkpoint,
        "cfg_text_scale": training_args.cfg_text_scale,
        "cfg_img_scale": training_args.cfg_img_scale,
        "timestep_shift": training_args.timestep_shift,
    }
    configured_weights = {
        key: getattr(training_args, f"{key}_weight")
        for key in LOSS_WEIGHTS
    }
    logger.info("E2E loss weights: %s", configured_weights)
    logger.info(
        "E2E schedule: num_timesteps=%s (%s Euler updates), gradient_steps=%s",
        training_args.num_timesteps,
        training_args.num_timesteps - 1,
        training_args.gradient_denoise_steps,
    )
    logger.info(
        "E2E execution: repair reason/score forward/backward, repair flow "
        "forward/backward, heatmap E2E forward/backward, then one optimizer "
        "step"
    )
    logger.info(
        "E2E VAE decoder activation checkpoint: %s",
        training_args.vae_decoder_checkpoint,
    )
    logger.info(
        "E2E phase-boundary CUDA cache release: %s",
        training_args.e2e_empty_cache,
    )
    logger.info(
        "E2E CPU gradient staging: %s",
        training_args.e2e_cpu_gradient_staging,
    )
    logger.info(
        "E2E repair-flow + heatmap activation CPU offload: %s",
        training_args.e2e_activation_cpu_offload,
    )

    data_status = data_status or {}
    optimizer.zero_grad()
    start_time = time()
    last_saved_step = None
    completed_step = train_step

    for micro_step, raw_data in enumerate(train_loader):
        completed_step = train_step + micro_step // training_args.gradient_accumulation_steps
        if completed_step >= training_args.total_steps:
            break
        data, data_indexes = _prepare_e2e_inputs(raw_data, vae_model, device)
        staged_gradients = {}

        # Phase 1: teacher-forced repair reason and score. Do not clear
        # gradients between phases: all three accumulate into the same
        # parameter gradients before one optimizer step.
        with torch.amp.autocast(
            "cuda", enabled=True, dtype=torch.bfloat16
        ):
            repair_reason_score_loss_dict = fsdp_model(
                e2e_inputs=data,
                e2e_vae_model=vae_model,
                e2e_options={**options, "phase": "repair_reason_score"},
            )
            repair_reason_score_total_loss = sum(
                repair_reason_score_loss_dict[name]
                * configured_weights[name]
                for name in REPAIR_REASON_SCORE_LOSS_NAMES
            )
            repair_reason_score_scaled_loss = (
                repair_reason_score_total_loss
                / training_args.gradient_accumulation_steps
            )
        repair_reason_score_scaled_loss.backward()
        loss_values = {
            name: value.detach()
            for name, value in repair_reason_score_loss_dict.items()
        }
        repair_reason_score_total_value = (
            repair_reason_score_total_loss.detach()
        )
        if training_args.e2e_cpu_gradient_staging:
            _stage_parameter_gradients(fsdp_model, staged_gradients)
        del (
            repair_reason_score_loss_dict,
            repair_reason_score_total_loss,
            repair_reason_score_scaled_loss,
        )
        _release_phase_cache(training_args.e2e_empty_cache)

        # Phase 2: rebuild the teacher-forced repair prefix and supervise only
        # flow matching. The phase-1 graph has already been released.
        repair_activation_offload_context = (
            torch.autograd.graph.save_on_cpu(
                pin_memory=True, device_type="cuda"
            )
            if training_args.e2e_activation_cpu_offload
            else nullcontext()
        )
        with torch.amp.autocast(
            "cuda", enabled=True, dtype=torch.bfloat16
        ), repair_activation_offload_context:
            repair_flow_loss_dict = fsdp_model(
                e2e_inputs=data,
                e2e_vae_model=vae_model,
                e2e_options={**options, "phase": "repair_flow"},
            )
            repair_flow_total_loss = sum(
                repair_flow_loss_dict[name] * configured_weights[name]
                for name in REPAIR_FLOW_LOSS_NAMES
            )
            repair_flow_scaled_loss = (
                repair_flow_total_loss
                / training_args.gradient_accumulation_steps
            )
        repair_flow_scaled_loss.backward()
        loss_values.update(
            {
                name: value.detach()
                for name, value in repair_flow_loss_dict.items()
            }
        )
        repair_flow_total_value = repair_flow_total_loss.detach()
        if training_args.e2e_cpu_gradient_staging:
            _stage_parameter_gradients(fsdp_model, staged_gradients)
        del (
            repair_flow_loss_dict,
            repair_flow_total_loss,
            repair_flow_scaled_loss,
        )
        _release_phase_cache(training_args.e2e_empty_cache)

        # Phase 3: construct a fresh Task-1 sampling -> differentiable VAE
        # decode -> ViT -> Task-2 heatmap graph. The heatmap losses remain
        # connected to the final K Task-1 denoising updates.
        activation_offload_context = (
            torch.autograd.graph.save_on_cpu(
                pin_memory=True, device_type="cuda"
            )
            if training_args.e2e_activation_cpu_offload
            else nullcontext()
        )
        with torch.amp.autocast(
            "cuda", enabled=True, dtype=torch.bfloat16
        ), activation_offload_context:
            heatmap_loss_dict = fsdp_model(
                e2e_inputs=data,
                e2e_vae_model=vae_model,
                e2e_options={**options, "phase": "heatmap"},
            )
            heatmap_total_loss = sum(
                heatmap_loss_dict[name] * configured_weights[name]
                for name in HEATMAP_LOSS_NAMES
            )
            heatmap_scaled_loss = (
                heatmap_total_loss
                / training_args.gradient_accumulation_steps
            )
        # The complete heatmap graph is still live here. empty_cache() does
        # not touch it or the gradients accumulated by phases 1 and 2; it
        # only returns fragmented, unused allocator blocks before backward's
        # FSDP all-gathers request their contiguous buffers.
        if training_args.e2e_empty_cache:
            torch.cuda.empty_cache()
        heatmap_scaled_loss.backward()
        loss_values.update(
            {
                name: value.detach()
                for name, value in heatmap_loss_dict.items()
            }
        )
        heatmap_total_value = heatmap_total_loss.detach()
        total_loss = (
            repair_reason_score_total_value
            + repair_flow_total_value
            + heatmap_total_value
        )
        del heatmap_loss_dict, heatmap_total_loss, heatmap_scaled_loss
        if training_args.e2e_cpu_gradient_staging:
            # Backward has released the heatmap graph. Return unused CUDA
            # blocks before restoring the smaller, sharded gradient buffers
            # one at a time and adding the repair gradients exactly once.
            gc.collect()
            torch.cuda.empty_cache()
            _restore_parameter_gradients(staged_gradients)

        optimizer_updated = (
            (micro_step + 1) % training_args.gradient_accumulation_steps == 0
        )
        if optimizer_updated:
            total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            if ema_model is not None:
                fsdp_ema_update(
                    ema_model, fsdp_model, decay=training_args.ema
                )
            optimizer.zero_grad()
        else:
            total_norm = total_loss.new_zeros(())

        quality = raw_data.get("quality", "unknown")
        if data_indexes is not None:
            dataset_name = data_indexes["dataset_name"]
            worker_id = data_indexes["worker_id"]
            data_status.setdefault(dataset_name, {})[worker_id] = data_indexes[
                "data_indexes"
            ]

        if completed_step % training_args.log_every == 0:
            elapsed = max(time() - start_time, 1e-6)
            log_values = {
                name: value.float() for name, value in loss_values.items()
            }
            log_values["total"] = total_loss.float()
            for value in log_values.values():
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                value.div_(dist.get_world_size())
            message = f"(step={completed_step:07d}, quality={quality}) "
            message += ", ".join(
                f"{name}={value.item():.4f}"
                for name, value in log_values.items()
            )
            message += f", steps/sec={training_args.log_every / elapsed:.3f}"
            logger.info(message)
            if dist.get_rank() == 0:
                print(message, flush=True)
                wandb.log(
                    {
                        **{
                            f"e2e/{name}": value.item()
                            for name, value in log_values.items()
                        },
                        "lr": optimizer.param_groups[0]["lr"],
                        "total_norm": total_norm.item(),
                        "quality_bad": float(quality == "bad"),
                        "mem_allocated": torch.cuda.max_memory_allocated()
                        / 1024**2,
                        "mem_cache": torch.cuda.max_memory_reserved() / 1024**2,
                    },
                    step=completed_step,
                )
            start_time = time()

        if (
            completed_step > 0
            and optimizer_updated
            and completed_step % training_args.save_every == 0
        ):
            _save_checkpoint(
                completed_step,
                fsdp_model,
                ema_model,
                optimizer,
                scheduler,
                logger,
                fsdp_config,
                training_args.checkpoint_dir,
                data_status,
            )
            last_saved_step = completed_step

    if completed_step > 0 and completed_step != last_saved_step:
        logger.info("Saving final E2E checkpoint at step %s", completed_step)
        _save_checkpoint(
            completed_step,
            fsdp_model,
            ema_model,
            optimizer,
            scheduler,
            logger,
            fsdp_config,
            training_args.checkpoint_dir,
            data_status,
        )

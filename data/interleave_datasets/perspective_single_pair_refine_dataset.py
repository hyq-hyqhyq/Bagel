# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Isolated perspective single/pair/refine multitask training dataset."""

import os
import re

from PIL import Image

from .reason_heatmap_dataset import ReasonHeatmapIterableDataset
from ..reason_heatmap_prompts import (
    PAIR_HEATMAP_PROMPT,
    REFINE_PROMPT,
    SINGLE_HEATMAP_PROMPT,
)


class PerspectiveSinglePairRefineIterableDataset(ReasonHeatmapIterableDataset):
    """Emit a configurable single-heatmap/pair-heatmap/refine task mix.

    The legacy behavior remains a 2:1:1 mix without reason supervision. New
    experiments can set ``BAGEL_PERSPECTIVE_MULTITASK_RATIO`` (for example,
    ``4:1:1``) and enable supervised reason text with
    ``BAGEL_PERSPECTIVE_MULTITASK_REASON=1``.
    """

    _RATIO_ENV = "BAGEL_PERSPECTIVE_MULTITASK_RATIO"
    _REASON_ENV = "BAGEL_PERSPECTIVE_MULTITASK_REASON"
    _DISABLE_HEATMAP_VISUAL_DROPOUT_ENV = (
        "BAGEL_PERSPECTIVE_DISABLE_HEATMAP_VISUAL_DROPOUT"
    )
    _USE_PAIR_REASON_ENV = "BAGEL_PERSPECTIVE_USE_PAIR_REASON"
    _JUDGMENT_ENV = "BAGEL_PERSPECTIVE_JUDGMENT"
    _ON_POLICY_JUDGMENT_ENV = "BAGEL_PERSPECTIVE_ON_POLICY_JUDGMENT"

    def __init__(self, *args, **kwargs):
        if "heatmap_only" in kwargs:
            raise ValueError(
                "PerspectiveSinglePairRefineIterableDataset does not support "
                "heatmap_only; its task recipe is fixed."
            )
        super().__init__(*args, heatmap_only=False, **kwargs)
        ratio = os.environ.get(self._RATIO_ENV, "2:1:1")
        try:
            task_ratio = tuple(int(value) for value in ratio.split(":"))
        except ValueError as exc:
            raise ValueError(
                f"{self._RATIO_ENV} must contain three integers, got {ratio!r}"
            ) from exc
        if len(task_ratio) != 3 or any(value < 0 for value in task_ratio):
            raise ValueError(
                f"{self._RATIO_ENV} must be SINGLE:PAIR:REFINE with three "
                f"non-negative integers, got {ratio!r}"
            )
        if sum(task_ratio) == 0:
            raise ValueError(f"{self._RATIO_ENV} cannot be 0:0:0")
        self.task_ratio = task_ratio

        reason_flag = os.environ.get(self._REASON_ENV, "0").strip().lower()
        if reason_flag not in {"0", "1", "false", "true", "no", "yes"}:
            raise ValueError(
                f"{self._REASON_ENV} must be a boolean, got {reason_flag!r}"
            )
        self.include_reason = reason_flag in {"1", "true", "yes"}

        disable_visual_dropout_flag = os.environ.get(
            self._DISABLE_HEATMAP_VISUAL_DROPOUT_ENV, "0"
        ).strip().lower()
        if disable_visual_dropout_flag not in {
            "0", "1", "false", "true", "no", "yes"
        }:
            raise ValueError(
                f"{self._DISABLE_HEATMAP_VISUAL_DROPOUT_ENV} must be a "
                f"boolean, got {disable_visual_dropout_flag!r}"
            )
        self.disable_heatmap_visual_dropout = (
            disable_visual_dropout_flag in {"1", "true", "yes"}
        )

        use_pair_reason_flag = os.environ.get(
            self._USE_PAIR_REASON_ENV, "0"
        ).strip().lower()
        if use_pair_reason_flag not in {
            "0", "1", "false", "true", "no", "yes"
        }:
            raise ValueError(
                f"{self._USE_PAIR_REASON_ENV} must be a boolean, got "
                f"{use_pair_reason_flag!r}"
            )
        self.use_pair_reason = use_pair_reason_flag in {"1", "true", "yes"}
        judgment_flag = os.environ.get(self._JUDGMENT_ENV, "1").strip().lower()
        if judgment_flag not in {"0", "1", "false", "true", "no", "yes"}:
            raise ValueError(f"{self._JUDGMENT_ENV} must be a boolean")
        self.include_judgment = judgment_flag in {"1", "true", "yes"}
        on_policy_flag = os.environ.get(
            self._ON_POLICY_JUDGMENT_ENV, "0"
        ).strip().lower()
        if on_policy_flag not in {"0", "1", "false", "true", "no", "yes"}:
            raise ValueError(
                f"{self._ON_POLICY_JUDGMENT_ENV} must be a boolean"
            )
        self.on_policy_judgment = on_policy_flag in {"1", "true", "yes"}
        if self.on_policy_judgment and not self.include_reason:
            raise ValueError(
                "On-policy judgment requires "
                f"{self._REASON_ENV}=1"
            )
        print(
            f"dataset-{self.dataset_name}: multitask_ratio="
            f"{':'.join(str(value) for value in self.task_ratio)}, "
            f"reason_supervision={self.include_reason}, "
            "disable_heatmap_visual_dropout="
            f"{self.disable_heatmap_visual_dropout}, "
            f"use_pair_reason={self.use_pair_reason}"
            f", judgment_supervision={self.include_judgment}"
            f", on_policy_judgment={self.on_policy_judgment}"
        )

    @staticmethod
    def _compact_reason(reason):
        """Remove answer-bearing inspection and conclusion fields.

        Judge metadata contains both structured dictionaries and the formatted
        text produced by ``prepare_perspective_strong_long_reason.py``.  Keep
        the neutral scene/structure/check specification in either case.  A
        legacy free-form reason with no named fields is kept as-is because it
        cannot be split safely.
        """
        if isinstance(reason, dict):
            lines = [
                "Scene:",
                str(reason.get("scene", "")).strip(),
                "",
                "Structures:",
            ]
            lines.extend(f"- {item}" for item in reason.get("structures", []))
            lines.extend(["", "Checks:"])
            for index, check in enumerate(reason.get("checks", []), start=1):
                lines.extend([f"Check {index}:", "Elements:"])
                lines.extend(f"- {item}" for item in check.get("elements", []))
                lines.extend(
                    [
                        "Expected relationship:",
                        str(check.get("expected_relationship", "")).strip(),
                        "",
                    ]
                )
            return "\n".join(lines).strip()

        text = str(reason or "").strip()
        if not text:
            return text
        lines = text.splitlines()
        has_named_fields = any(
            re.match(r"^\s*(?:Inspection|Conclusion)\s*:\s*$", line, re.I)
            for line in lines
        )
        if not has_named_fields:
            return text

        kept = []
        skipping_inspection = False
        for line in lines:
            stripped = line.strip()
            if re.match(r"^Conclusion\s*:\s*$", stripped, re.I):
                break
            if re.match(r"^Inspection\s*:\s*$", stripped, re.I):
                skipping_inspection = True
                continue
            if skipping_inspection:
                if re.match(r"^Check\s+\d+\s*:\s*$", stripped, re.I):
                    skipping_inspection = False
                else:
                    continue
            kept.append(line)

        # Avoid accumulating large runs of empty lines after removing fields.
        compact = []
        for line in kept:
            if not line.strip() and compact and not compact[-1].strip():
                continue
            compact.append(line)
        result = "\n".join(compact).strip()
        return result or text

    @staticmethod
    def _judgment_values(row, reason_key, quality):
        judge = row.get("judge") or {}
        checks = (judge.get("checks") or {}).get(reason_key)
        if not isinstance(checks, list):
            return None
        global_value = (judge.get("global") or {}).get(quality)
        if global_value is None:
            global_value = 1 if quality == "good" else 0
        return [int(value) for value in checks], int(global_value)

    @staticmethod
    def _judgment_text(row, reason_key, quality):
        """Render check and global polarity targets separately."""
        values = PerspectiveSinglePairRefineIterableDataset._judgment_values(
            row, reason_key, quality
        )
        if values is None:
            return None
        checks, global_value = values
        labels = []
        for index, value in enumerate(checks, start=1):
            label = "correct" if int(value) == 1 else "incorrect"
            labels.append(f"<judgment>Check {index} conclusion: {label}.</judgment>")
        global_label = "correct" if int(global_value) == 1 else "incorrect"
        return labels, f"<judgment>Global conclusion: {global_label}.</judgment>"

    def parse_row(self, row, data_dir):
        good_image = self._read_image(os.path.join(data_dir, row["good_image"]))
        bad_image = self._read_image(os.path.join(data_dir, row["bad_image"]))
        bad_heatmap = self._read_image(
            os.path.join(data_dir, row["bad_heatmap"])
        )
        black_heatmap = Image.new("RGB", good_image.size)

        single_specs = [
            (
                "single_good_heatmap",
                "heatmap",
                "good",
                [good_image],
                SINGLE_HEATMAP_PROMPT,
                black_heatmap,
                row.get("good_score", 1.0),
            ),
            (
                "single_bad_heatmap",
                "heatmap",
                "bad",
                [bad_image],
                SINGLE_HEATMAP_PROMPT,
                bad_heatmap,
                row.get("bad_score", 0.0),
            ),
        ]
        # For pair tasks the first image is always the original and the second
        # image is always its refined reference. The target mask is aligned
        # with the first image.
        pair_specs = [
            (
                "pair_good_heatmap",
                "heatmap",
                "good",
                [good_image, good_image],
                PAIR_HEATMAP_PROMPT,
                black_heatmap,
                row.get("good_score", 1.0),
            ),
            (
                "pair_bad_heatmap",
                "heatmap",
                "bad",
                [bad_image, good_image],
                PAIR_HEATMAP_PROMPT,
                bad_heatmap,
                row.get("bad_score", 0.0),
            ),
        ]
        refine_specs = [
            (
                "good_refine",
                "repair",
                "good",
                [good_image],
                REFINE_PROMPT,
                good_image,
                row.get("good_score", 1.0),
            ),
            (
                "bad_refine",
                "repair",
                "bad",
                [bad_image],
                REFINE_PROMPT,
                good_image,
                row.get("bad_score", 0.0),
            ),
        ]
        single_weight, pair_weight, refine_weight = self.task_ratio
        task_specs = (
            single_specs * single_weight
            + pair_specs * pair_weight
            + refine_specs * refine_weight
        )

        samples = []
        for (
            task_name,
            gen_task,
            quality,
            input_images,
            prompt,
            target_image,
            score_label,
        ) in task_specs:
            data = self._init_data()
            enable_visual_cfg = not (
                gen_task == "heatmap"
                and self.disable_heatmap_visual_dropout
            )
            for input_image in input_images:
                data = self._add_image(
                    data,
                    input_image,
                    need_loss=False,
                    need_vae=True,
                    need_vit=True,
                    enable_cfg=enable_visual_cfg,
                )
            data = self._add_text(
                data,
                prompt,
                need_loss=False,
                enable_cfg=False,
            )
            if self.include_reason:
                reason_key = (
                    "pair_reason"
                    if self.use_pair_reason
                    and task_name == "pair_bad_heatmap"
                    else f"{quality}_reason"
                )
                reason = row[reason_key]
                on_policy = getattr(self, "on_policy_judgment", False)
                if on_policy:
                    reason = self._compact_reason(reason)
                data = self._add_text(
                    data,
                    f"<think>{reason}</think>",
                    need_loss=True,
                    enable_cfg=False,
                    loss_type="reason",
                )
                if getattr(self, "include_judgment", False):
                    if on_policy:
                        judgment = self._judgment_values(
                            row, reason_key, quality
                        )
                        if judgment is not None:
                            checks, global_value = judgment
                            # image_tensor_list currently contains one VAE/VIT
                            # pair per conditioning image; target image tensors
                            # are appended below.
                            input_tensors = data["image_tensor_list"]
                            data["judgment_rollout"] = {
                                "vae_images": list(input_tensors[0::2]),
                                "vit_images": list(input_tensors[1::2]),
                                "prompt_ids": self.tokenizer.encode(prompt),
                                "check_labels": checks,
                                "global_label": global_value,
                                "gen_task": gen_task,
                                "task_name": task_name,
                            }
                    else:
                        judgment = self._judgment_text(
                            row, reason_key, quality
                        )
                        if judgment is not None:
                            checks, global_text = judgment
                            for check_text in checks:
                                data = self._add_text(
                                    data, check_text, need_loss=True,
                                    enable_cfg=False,
                                    loss_type="judgment",
                                )
                            data = self._add_text(
                                data,
                                global_text,
                                need_loss=True,
                                enable_cfg=False,
                                loss_type="global",
                            )
            data = self._add_image(
                data,
                target_image,
                need_loss=True,
                need_vae=False,
                need_vit=False,
            )
            if score_label is not None:
                data["score_label"] = float(score_label)
            data["gen_task"] = gen_task
            data["gen_quality"] = quality
            data["task_name"] = task_name
            samples.append(data)

        return samples

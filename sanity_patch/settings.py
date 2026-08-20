# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

SANITY_PATCH_PROMPT = (
    "Detect the artificial rectangular patch and output a binary heatmap: "
    "white for the patch and black for the background. If no patch is present, "
    "output an all-black heatmap."
)

TEXT_COND_DROPOUT_PROB = 0.05
VAE_COND_DROPOUT_PROB = 0.1
VIT_COND_DROPOUT_PROB = 0.1
TIMESTEP_SHIFT = 4.0
BINARY_MASK_THRESHOLD = 127

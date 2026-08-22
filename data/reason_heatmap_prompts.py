# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


PERSPECTIVE_REFINE_PROMPT = (
    "Analyze the perspective and projection realism of the input image, explain "
    "any issue, assess its quality, and generate a refined image that corrects "
    "the issue while preserving unaffected content. If no correction is needed, "
    "reproduce the input image unchanged."
)

PERSPECTIVE_VERIFY_PROMPT = (
    "The first image is the original and the second is its refined version. "
    "Compare them, explain any perspective or projection issue in the first "
    "image, assess its quality, and generate a binary correction heatmap aligned "
    "with the first image: white for regions requiring correction and black "
    "elsewhere. Output an all-black heatmap if no correction is needed."
)

SANITY_REFINE_PROMPT = (
    "Analyze whether the input image contains an artificial rectangular patch, "
    "explain your judgment, assess its quality, and generate a refined image "
    "with the patch removed while preserving unaffected content. If no patch is "
    "present, reproduce the input image unchanged."
)

SANITY_VERIFY_PROMPT = (
    "The first image is the original and the second is its refined version. "
    "Compare them, explain whether the first image contains an artificial "
    "rectangular patch, assess its quality, and generate a binary heatmap: white "
    "for the patch and black for the background. Output an all-black heatmap if "
    "no patch is present."
)

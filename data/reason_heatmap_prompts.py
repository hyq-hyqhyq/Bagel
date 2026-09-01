# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


GEN_THINK_SYSTEM_PROMPT = (
    "You should first think about the planning process in the mind and then "
    "generate the image. \n"
    "The planning process is enclosed within <think> </think> tags, i.e. "
    "<think> planning process here </think> image here"
)


SINGLE_HEATMAP_PROMPT = """Analyze the input image for perspective or projection inconsistencies.

Generate a binary localization mask of the erroneous region in the input image.
Use white for perspective/projection errors and black for all other regions.

If no perspective/projection error is present, output an entirely black mask."""

PAIR_HEATMAP_PROMPT = """Compare the original image with the refined reference image.

Identify the perspective or projection inconsistency present in the original image and generate a binary localization mask on the original image.
Use white for the erroneous region and black for all other regions.

If the original image contains no perspective/projection error, output an entirely black mask."""

REFINE_PROMPT = """Correct any perspective or projection inconsistency in the input image while preserving all unrelated content and appearance.

Preserve object identity, scene semantics, materials, textures, colors, lighting, and overall photorealism.
Keep the original camera viewpoint and framing.

If no correction is necessary, preserve the input image unchanged."""


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

# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from PIL import Image

from .settings import BINARY_MASK_THRESHOLD


def to_binary_mask(image, threshold=BINARY_MASK_THRESHOLD):
    grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    binary = np.where(grayscale > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary)

import sys
from pathlib import Path

import cv2
import pytest
import numpy as np

current_dir = Path(__file__).resolve().parent
sys.path.append(current_dir.parent.as_posix())

from solver.core.transforms import get_transforms


@pytest.mark.parametrize(
    "mode",
    ["train", "validation", "test"]
)
def test_transforms(mode):
    size = (28, 28)
    images = np.random.randint(0, 255, (*size, 3)).astype(np.uint8)

    transforms = get_transforms(size=size)[mode]
    transformed = transforms(images)

    assert transformed.shape == (1, *size)

import sys
from pathlib import Path

import pytest
import torch

current_dir = Path(__file__).resolve().parent
sys.path.append(current_dir.parent.as_posix())

from solver.model import get_resnet


@pytest.mark.parametrize(
    "batch_size, ckpt", [(4, "assets/ckpt/best_ckpt.pth")]
)
def test_model(batch_size, ckpt):
    size = (512, 512)
    images = torch.rand(size=(batch_size, 1, *size))

    model = get_resnet(num_classes=10, pretrained=False)
    model.load_state_dict(torch.load(ckpt))

    out = model(images)

    assert out.shape == (batch_size, 10)

from pathlib import Path
from typing import List

import cv2
import torch

from solver.model import get_resnet

from .transforms import get_transforms


class Detector:
    def __init__(self, ckpt: str = "../assets/ckpt/best_ckpt.pth") -> None:
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.model = get_resnet(num_classes=10, pretrained=False)
        self.model.load_state_dict(
            torch.load(ckpt, map_location=torch.device(self.device))
        )
        self.transforms = get_transforms()
        self.table = [[0 for _ in range(9)] for _ in range(9)]

    def detect(
        self, image_path: str = "../assets/data/problem/example"
    ) -> List:
        self.model.eval()

        image_path = Path(image_path)
        dir = image_path.parent / image_path.stem
        for p in dir.glob("**/*.jpg"):
            column, row = map(int, str(p.stem).strip(""))

            img = cv2.imread(p.as_posix())
            img = self.transforms["test"](img).view(1, 1, 28, 28)

            self.table[column][row] = self.model(img).argmax().item()

        return self.table

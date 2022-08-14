from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from solver.model import get_resnet

from .transforms import get_transforms


class Detector:
    def __init__(
        self, ckpt: str = "../logs/NumberPlaceDataset/ckpt/best_ckpt.pth"
    ) -> None:
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.model = get_resnet(pretrained=False, num_classes=10)
        self.model.load_state_dict(
            torch.load(ckpt, map_location=torch.device(self.device))
        )
        self.transforms = get_transforms()
        self.table = [[0 for _ in range(9)] for _ in range(9)]

    def detect(
        self,
        image_path: str = "../assets/data/problem/example"
    ) -> List:
        self.model.eval()

        image_path = Path(image_path)
        dir = image_path.parent / image_path.stem
        for p in dir.glob("**/*.jpg"):
            column, row = map(int, str(p.stem).strip(""))

            gray = Image.open(p).convert("L")
            gray = ImageOps.invert(gray)
            gray = np.array(gray)
            ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
            th = Image.fromarray(th)
            img = self.transforms["test"](th).view(1, 1, 28, 28)

            self.table[column][row] = self.model(img).argmax().item()

        return self.table

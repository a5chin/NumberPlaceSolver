import numpy as np
import cv2
from torchvision.datasets import ImageFolder
from PIL import Image, ImageOps
from random import randint
from typing import Any, Callable, Optional, Tuple


def draw_rect(img: np.array, eps: int=2) -> np.array:
    rands = [randint(-eps, eps) for _ in range(4)]
    height, width, _ = img.shape
    cv2.rectangle(img, (0 + rands[0], 0 + rands[1]), (width + rands[2], height + rands[3]), (0, 0, 0))
    return img


class NumberPlaceDataset(ImageFolder):
    def __init__(
        self,
        root: str='../data/NumberPlaceDataset',
        transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transform)
        self.root = root
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)

        img = np.array(sample)
        img = draw_rect(img)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        th = Image.fromarray(th)
        th = ImageOps.invert(th)

        if self.transform is not None:
            img = self.transform(th)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

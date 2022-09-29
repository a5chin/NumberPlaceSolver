from random import randint
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms


class Color2Bin:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        return th


class RandomDrawRect:
    def __init__(self, eps: int = 2) -> None:
        self.eps = eps

    def __call__(self, img: np.ndarray) -> np.ndarray:
        rands = [randint(-self.eps, self.eps) for _ in range(4)]
        height, width = img.shape
        cv2.rectangle(
            img,
            (0 + rands[0], 0 + rands[1]),
            (width + rands[2], height + rands[3]),
            (0, 0, 0),
        )

        return img


class Invert:
    def __call__(self, th: np.ndarray) -> np.ndarray:
        th = cv2.bitwise_not(th)

        return th


class GaussianNoise:
    def __init__(self, mean=0.0, std=1.0) -> None:
        self.std = std
        self.mean = mean

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


def get_transforms(size: Tuple[int] = (28, 28)) -> Dict:
    return {
        "train": transforms.Compose(
            [
                Color2Bin(),
                RandomDrawRect(),
                Invert(),
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(
                    size=size,
                    scale=(0.08, 1.0),
                    ratio=(3 / 4, 4 / 3),
                ),
                transforms.RandomRotation(degrees=10),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.ToTensor(),
                GaussianNoise(mean=0.0, std=0.08),
            ]
        ),
        "validation": transforms.Compose(
            [
                Color2Bin(),
                Invert(),
                transforms.ToPILImage(),
                transforms.Resize(size=size),
                transforms.ToTensor(),
            ]
        ),
        "test": transforms.Compose(
            [
                Color2Bin(),
                Invert(),
                transforms.ToPILImage(),
                transforms.Resize(size=size),
                transforms.ToTensor(),
            ]
        ),
    }

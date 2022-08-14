from typing import Tuple

from torch import randn
from torchvision import transforms


class GaussianNoise:
    def __init__(self, mean=0.0, std=1.0) -> None:
        self.std = std
        self.mean = mean

    def __call__(self, tensor) -> float:
        return tensor + randn(tensor.size()) * self.std + self.mean

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


def get_transforms(size: Tuple[int] = (28, 28)) -> dict:
    return {
        "train": transforms.Compose(
            [
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
            [transforms.Resize(size=size), transforms.ToTensor()]
        ),
        "test": transforms.Compose(
            [transforms.Resize(size=size), transforms.ToTensor()]
        ),
    }

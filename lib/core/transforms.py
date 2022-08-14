from torch import randn
from torchvision import transforms

from lib.config import config


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


def get_transforms() -> dict:
    return {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    config.MODEL.INPUT_SIZE,
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
            [transforms.Resize(config.MODEL.INPUT_SIZE), transforms.ToTensor()]
        ),
        "test": transforms.Compose(
            [transforms.Resize(config.MODEL.INPUT_SIZE), transforms.ToTensor()]
        ),
    }

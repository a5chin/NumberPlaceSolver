from typing import Any, Callable, Optional, Tuple

import numpy as np
from torchvision.datasets import ImageFolder


class NumberPlaceDataset(ImageFolder):
    def __init__(
        self,
        root: str = "../assets/data/NumberPlaceDataset",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)

        img = np.array(sample)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

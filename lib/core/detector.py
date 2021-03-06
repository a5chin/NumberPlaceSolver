import torch
import numpy as np
import cv2
from PIL import Image, ImageOps
from pathlib import Path
from typing import List

from lib.core import get_transforms
from lib.model import get_resnet


class Detector:
    def __init__(self, ckpt: str='../logs/NumberPlaceDataset/ckpt/best_ckpt.pth') -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = get_resnet(pretrained=False, num_classes=10)
        self.model.load_state_dict(torch.load(ckpt, map_location=torch.device(self.device)))
        self.transforms = get_transforms()
        self.table = [[0 for _ in range(9)] for _ in range(9)]

    def detect(self, dir: str='../data/problem/example') -> List:
        self.model.eval()
        dir = Path(dir)
        for p in dir.glob('**/*.jpg'):
            column, row = map(int, str(p.stem).strip(''))

            gray = Image.open(p).convert('L')
            gray = ImageOps.invert(gray)
            gray = np.array(gray)
            ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
            th = Image.fromarray(th)
            img = self.transforms['test'](th).view(1, 1, 28, 28)

            self.table[column][row] = self.model(img).argmax().item()

        return self.table

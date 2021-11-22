import cv2
from pathlib import Path


class CutOuter:
    def __init__(self, root: str='../data/problem', name: str='example.png'):
        self.root = Path(root)
        self.images_path = self.root / name
        self.temp = self.root / self.images_path.stem
        self.img = None

    def cutout(self, eps=0):
        self.temp.mkdir(exist_ok=True)
        img = cv2.imread(str(self.images_path), cv2.IMREAD_GRAYSCALE)
        self.img = cv2.resize(img, dsize=(28 * 9, 28 * 9))
        height, width = self.img.shape
        cru = height // 9
        for y in range(9):
            for x in range(9):
                temp = self.img[cru * x + eps: cru * (x + 1) - eps, cru * y + eps: cru * (y + 1) - eps]
                cv2.imwrite(str(self.temp / f'{x}{y}.jpg'), temp)

import numpy as np
import cv2
from pathlib import Path

from lib.config import config


class Reshaper:
    LENGTH = config.MODEL.INPUT_SIZE[0]

    def __init__(self, image_path: str = "data/problem/example2.png") -> None:
        self.point = np.array(
            [
                [Reshaper.LENGTH * 9, 0],
                [0, 0],
                [0, Reshaper.LENGTH * 9],
                [Reshaper.LENGTH * 9, Reshaper.LENGTH * 9],
            ],
            dtype=np.float32,
        )
        self.image_path = Path(image_path)
        self.th = Reshaper._load_image(image_path)
        self.square = self._get_square()

    def reshape(self) -> np.ndarray:
        mat = cv2.getPerspectiveTransform(self.square, self.point)
        image = cv2.warpPerspective(
            self.th, mat, (Reshaper.LENGTH * 9, Reshaper.LENGTH * 9)
        )
        return image

    def _get_square(self) -> np.ndarray:
        max_area = 0
        inv = cv2.bitwise_not(self.th)
        contours, hierarchy = cv2.findContours(
            inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            arclen = cv2.arcLength(cnt, True)
            approx_cnt = cv2.approxPolyDP(
                cnt, epsilon=0.001 * arclen, closed=True
            )
            if len(approx_cnt) == 4:
                area = cv2.contourArea(approx_cnt)
                if area > max_area:
                    max_area = max(area, max_area)
                    contour = approx_cnt
        return contour.astype(np.float32)

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        ret, th = cv2.threshold(
            cv2.imread(path, cv2.IMREAD_GRAYSCALE), 0, 255, cv2.THRESH_OTSU
        )
        cv2.imshow("input", th)
        return th

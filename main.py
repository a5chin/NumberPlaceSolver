import cv2
from lib.core import *


def main():
    cutouter = CutOuter(root='./data/problem', name='example.png')
    cutouter.cutout(eps=0)

    detector = Detector(ckpt='./logs/NumberPlaceDataset/ckpt/last_ckpt.pth')
    data = detector.detect(dir='./data/problem/example')

    solver = Solver()
    result = solver.get_result(data)

    img = cutouter.img
    height, width = img.shape
    
    cv2.imshow('raw', img)

    for i, col in enumerate(data):
        for j, item in enumerate(col):
            if int(item) == 0:
                cv2.putText(
                    img,
                    text=str(result[i][j]),
                    org=(width // 9 * j + 9, height // 9 * (i + 1) - 9),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0),
                    thickness=2,
                    lineType=cv2.LINE_4
                )

    cv2.imshow('result', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()

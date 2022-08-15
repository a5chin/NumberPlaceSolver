import argparse
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(current_dir.as_posix() + "/../")

import cv2

from solver.core import CutOuter, Detector, Solver


def make_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_path",
        default="../assets/data/problem/example2.png",
        type=str,
        help="plese set image path for inference",
    )
    parser.add_argument(
        "--ckpt",
        default="../assets/ckpt/last_ckpt.pth",
        type=str,
        help="plese set ckpt",
    )
    parser.add_argument(
        "--size",
        default=28,
        type=int,
        help="plese set image size",
    )

    return parser.parse_args()


def main():
    args = make_parse()

    cutouter = CutOuter(args=args)
    cutouter.cutout(eps=0)

    detector = Detector(ckpt=args.ckpt)
    data = detector.detect(image_path=args.image_path)

    solver = Solver()
    result = solver.get_result(data)

    img = cutouter.img
    height, width = img.shape

    raw = cv2.imread(args.image_path)
    cv2.imshow("raw", raw)

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
                    lineType=cv2.LINE_4,
                )

    cv2.imshow("result", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()

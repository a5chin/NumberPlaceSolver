import argparse
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
<<<<<<< HEAD
sys.path.append(current_dir.as_posix() + "/../")
=======
sys.path.append(current_dir.parent.as_posix())
>>>>>>> 149f6c930d176f05f6dd7da1624b4a011fec2e3f

from solver.core import Detector


def make_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
<<<<<<< HEAD
        default="../assets/ckpt/best_ckpt.pth",
        type=str,
        help="plese set ckpt",
    )
=======
        default="../assets/ckpt/last_ckpt.pth",
        type=str,
        help="plese set ckpt",
    )
    parser.add_argument(
        "--image",
        default="../assets/data/problem/example2",
        type=str,
        help="plese set path of problem image",
    )
>>>>>>> 149f6c930d176f05f6dd7da1624b4a011fec2e3f

    return parser.parse_args()


def main():
    args = make_parse()

    detector = Detector(ckpt=args.ckpt)
<<<<<<< HEAD
    result = detector.detect()
=======
    result = detector.detect(image_path=args.image)
>>>>>>> 149f6c930d176f05f6dd7da1624b4a011fec2e3f
    for res in result:
        print(*res)


if __name__ == "__main__":
    main()

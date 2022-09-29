import argparse
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(current_dir.parent.as_posix())

from solver.core import Detector


def make_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
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

    return parser.parse_args()


def main():
    args = make_parse()

    detector = Detector(ckpt=args.ckpt)
    result = detector.detect(image_path=args.image)
    for res in result:
        print(*res)


if __name__ == "__main__":
    main()

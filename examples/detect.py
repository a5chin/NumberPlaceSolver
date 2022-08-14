import argparse
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(current_dir.as_posix() + "/../")

from solver.core import Detector


def make_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        default="../logs/NumberPlaceDataset/ckpt/best_ckpt.pth",
        type=str,
        help="plese set ckpt",
    )

    return parser.parse_args()


def main():
    args = make_parse()

    detector = Detector(ckpt=args.ckpt)
    result = detector.detect()
    for res in result:
        print(*res)


if __name__ == "__main__":
    main()

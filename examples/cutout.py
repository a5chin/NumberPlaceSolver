import argparse
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(current_dir.parent.as_posix())

from solver.core import CutOuter


def make_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        default="../assets/data/problem/example2.png",
        type=str,
        help="plese set image path",
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
    image_path = args.image
    size = args.size

    cutouter = CutOuter(image_path=image_path, size=size)
    cutouter.cutout(eps=0)


if __name__ == "__main__":
    main()

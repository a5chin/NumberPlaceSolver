import argparse
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
<<<<<<< HEAD
sys.path.append(current_dir.as_posix() + "/../")
=======
sys.path.append(current_dir.parent.as_posix())
>>>>>>> 149f6c930d176f05f6dd7da1624b4a011fec2e3f

from solver.core import CutOuter


def make_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
<<<<<<< HEAD
        "--image_path",
=======
        "--image",
>>>>>>> 149f6c930d176f05f6dd7da1624b4a011fec2e3f
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
<<<<<<< HEAD

    cutouter = CutOuter(args=args)
=======
    image_path = args.image
    size = args.size

    cutouter = CutOuter(image_path=image_path, size=size)
>>>>>>> 149f6c930d176f05f6dd7da1624b4a011fec2e3f
    cutouter.cutout(eps=0)


if __name__ == "__main__":
    main()

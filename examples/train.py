import argparse
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(current_dir.parent.as_posix())

from solver.core import Trainer


def make_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="../assets/data/NumberPlaceDataset",
        type=str,
        help="plese set data root",
    )
    parser.add_argument(
        "--num_classes",
        default=10,
        type=int,
        help="plese set num_classes",
    )
    parser.add_argument(
        "--epoch",
        default=10000,
        type=int,
        help="plese set train epoch",
    )
    parser.add_argument(
        "--size",
        default=4,
        type=int,
        help="plese set image size",
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="plese set batch size",
    )
    parser.add_argument(
        "--logdir",
        default="../logs",
        type=str,
        help="plese set logdir",
    )

    return parser.parse_args()


def main():
    args = make_parse()

    trainer = Trainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()

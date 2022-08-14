from lib.core import Detector


def main():
    detector = Detector(ckpt="../logs/NumberPlaceDataset/ckpt/best_ckpt.pth")
    result = detector.detect()
    for res in result:
        print(*res)


if __name__ == "__main__":
    main()

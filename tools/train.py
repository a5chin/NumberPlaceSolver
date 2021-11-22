from lib.core import Trainer
from lib.config import load_config

cfg = load_config('/Users/a5/PycharmProjects/NumberPlace/experiments/resnet.yaml')


def main():
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()

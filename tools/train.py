from lib.core import Trainer
from lib.config import load_config

cfg = load_config('experiments/resnet.yaml')


def main():
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()

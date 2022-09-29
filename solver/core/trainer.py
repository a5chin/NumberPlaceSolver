from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
from timm import scheduler
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from solver.dataset import NumberPlaceDataset
from solver.model import get_resnet
from solver.utils import AverageMeter, get_logger

from .transforms import get_transforms


class Trainer:
    def __init__(self, args) -> None:
        self.args = args
        self.logger = get_logger()
        self.root = Path(args.root).expanduser()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.model = get_resnet(num_classes=args.num_classes, pretrained=True)
        self.transforms = get_transforms(size=(args.size, args.size))
        self.best_acc = 0.0
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3)
        self.scheduler = scheduler.CosineLRScheduler(
            optimizer=self.optimizer,
            t_initial=self.args.epoch,
            lr_min=1e-6,
            warmup_t=self.args.epoch / 10,
            warmup_lr_init=1e-7,
            warmup_prefix=True,
        )
        self.log_dir = f"{self.args.logdir}/{self.root.name}"
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def fit(self) -> None:
        traindataset = NumberPlaceDataset(
            root=self.root, transform=self.transforms["train"]
        )
        traindataloader = DataLoader(
            dataset=traindataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
        )

        for epoch in range(self.args.epoch):
            self.model.train()

            losses = AverageMeter("train_loss")
            accuracies = AverageMeter("train_acc")

            with tqdm(traindataloader) as pbar:
                pbar.set_description(
                    "[Epoch %d/%d]" % (epoch + 1, self.args.epoch)
                )

                for images, labels in pbar:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    preds = self.model(images)
                    loss = self.criterion(preds, labels)
                    loss.backward()

                    self.optimizer.step()

                    results = preds.cpu().detach().argmax(dim=1)
                    accuracies.update(
                        (results == labels).float().mean().item()
                    )
                    losses.update(loss.item())

                    pbar.set_postfix(
                        OrderedDict(Loss=losses.value, Acc=accuracies.value)
                    )

            torch.save(
                self.model.state_dict(), self.log_dir / Path("last_ckpt.pth")
            )

            self.evaluate(self.model, epoch + 1)

            lr = self.scheduler.get_epoch_values(epoch)[0]
            self.writer.add_scalar("lr", lr, epoch + 1)
            self.scheduler.step(epoch + 1)

    @torch.inference_mode()
    def evaluate(self, model: nn.Module, epoch: Optional[int] = None) -> None:
        model.eval()

        losses = AverageMeter("valid_loss")
        accuracies = AverageMeter("valid_acc")

        valdataset = NumberPlaceDataset(
            root=self.root, transform=self.transforms["validation"]
        )
        valdataloader = DataLoader(
            dataset=valdataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=True,
        )

        for data in valdataloader:
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)

            preds = model(images)
            loss = self.criterion(preds, labels)

            results = preds.cpu().detach().argmax(dim=1)
            accuracies.update((results == labels).float().mean().item())
            losses.update(loss.item())

        self.logger.info(f"Loss: {losses.avg}, Accuracy: {accuracies.avg}")

        if epoch is not None:
            self.writer.add_scalar("loss", losses.avg, epoch + 1)
            self.writer.add_scalar("accuracy", accuracies.avg, epoch + 1)
            if self.best_acc <= accuracies.avg:
                self.best_acc = accuracies.avg
                torch.save(
                    model.state_dict(), self.log_dir / Path("best_ckpt.pth")
                )

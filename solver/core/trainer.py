from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
from sklearn.metrics import accuracy_score
from timm import scheduler
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from solver.dataset import NumberPlaceDataset
from solver.model import get_resnet

from .transforms import get_transforms


class Trainer:
    def __init__(self, args) -> None:
        self.args = args
        self.root = Path(args.root).expanduser()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.model = get_resnet(num_classes=args.num_classes)
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
            total, running_loss, running_acc = 0, 0.0, 0.0

            with tqdm(traindataloader) as pbar:
                pbar.set_description(
                    "[Epoch %d/%d]" % (epoch + 1, self.args.epoch)
                )

                for images, labels in pbar:
                    images, labels = images.to(self.device), labels.to(
                        self.device
                    )

                    self.optimizer.zero_grad()

                    preds = self.model(images)
                    loss = self.criterion(preds, labels)
                    loss.backward()

                    self.optimizer.step()

                    results = preds.cpu().detach().numpy().argmax(axis=1)
                    running_acc += accuracy_score(
                        labels.cpu().numpy(), results
                    ) * len(images)
                    running_loss += loss.item() * len(images)

                    total += len(images)

                    pbar.set_postfix(
                        OrderedDict(
                            Loss=running_loss / total, Acc=running_acc / total
                        )
                    )

                running_acc /= total
                running_loss /= total

            torch.save(
                self.model.state_dict(), self.log_dir / Path("last_ckpt.pth")
            )

            self.evaluate(self.model, epoch + 1)

            lr = self.scheduler.get_epoch_values(epoch)[0]
            self.writer.add_scalar("lr", lr, epoch + 1)
            self.scheduler.step(epoch + 1)

    def evaluate(
        self, model: Optional[nn.Module], epoch: Optional[int] = None
    ) -> None:
        valdataset = NumberPlaceDataset(
            root=self.root, transform=self.transforms["validation"]
        )
        valdataloader = DataLoader(
            dataset=valdataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=True,
        )

        model.eval()
        total, val_loss, val_acc = 0, 0.0, 0.0
        with torch.inference_mode():
            for data in valdataloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                preds = model(images)
                loss = self.criterion(preds, labels)
                results = preds.cpu().detach().numpy().argmax(axis=1)
                val_acc += accuracy_score(labels.cpu().numpy(), results) * len(
                    labels
                )
                val_loss += loss.item() * len(labels)

                total += len(labels)

            val_acc /= total
            val_loss /= total

        print("Loss: %f, Accuracy: %f" % (val_loss, val_acc))

        if epoch is not None:
            self.writer.add_scalar("loss", val_loss, epoch + 1)
            self.writer.add_scalar("accuracy", val_acc, epoch + 1)
            if self.best_acc <= val_acc:
                self.best_acc = val_acc
                torch.save(
                    model.state_dict(), self.log_dir / Path("best_ckpt.pth")
                )

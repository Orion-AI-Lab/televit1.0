from typing import Any, List

import torch
from lightning.pytorch import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import AUROC, AveragePrecision, F1Score
import segmentation_models_pytorch as smp
import lightning.pytorch as pl
from transformers import get_cosine_schedule_with_warmup


class plUNET(pl.LightningModule):
    def __init__(
            self,
            input_vars: list = None,
            positional_vars: list = None,
            lr: float = 0.001,
            weight_decay: float = 0.0005,
            loss='ce',
            encoder='efficientnet-b1',
            warmup_ratio=0.05,
            sea_masked=False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = smp.UnetPlusPlus(encoder_name=encoder, in_channels=len(input_vars) + len(positional_vars), classes=2)
        if loss == 'dice':
            self.criterion = smp.losses.DiceLoss(mode='multiclass')
        elif loss == 'ce':
            self.criterion = torch.nn.CrossEntropyLoss()

        self.sea_masked = sea_masked

        if self.sea_masked:
            self.val_auprc = AveragePrecision(num_classes=1,  task="binary", ignore_index=2)
            self.test_auprc = AveragePrecision(num_classes=1, task="binary", ignore_index=2)
        else:
            self.val_auprc = AveragePrecision(task='binary', num_classes=1)
            self.test_auprc = AveragePrecision(task='binary', num_classes=1)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        # TODO remove squeeze once the model is made to handle inputs of shape (c, t, h, w)
        x_local = batch['x_local'].squeeze().float()
        x_local_pos = batch['x_local_pos'].float()

        x = torch.cat([x_local, x_local_pos], dim=1)
        y_local = batch['y_local']
        y = y_local.long()
        # if this is the first batch
        if self.global_step == 0:
            # print the shapes of the inputs and outputs
            print(f'x_local shape: {x_local.shape}')
            print(f'x_local_pos shape: {x_local_pos.shape}')
            print(f'x shape: {x.shape}')


        # calculate pad_size for x_local so that it is divisible by 32
        pad_size = (x_local.shape[2] % 32) // 2
    

        x = x.float()
        # pad x of shape (batch_size, C, 80, 80) to (batch_size, 1, 96, 96)
        if pad_size > 0:
            x = torch.nn.functional.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
            # pad y of shape (batch_size, 80, 80) to (batch_size, 96, 96)
            y = torch.nn.functional.pad(y, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)

        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.nn.functional.softmax(logits, dim=1)[:, 1]
        return loss, preds, y, x

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.step(batch)

        self.log("train/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True)

        return {"loss": loss}


    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.step(batch)
        # log val metrics
        self.val_auprc.update(preds.flatten(), targets.flatten())
        self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds.detach().cpu(), "targets": targets.detach().cpu(),
                "inputs": inputs.detach().cpu()}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.step(batch)

        self.test_auprc.update(preds, targets)
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds.detach().cpu(), "targets": targets.detach().cpu()}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        num_epochs = self.trainer.max_epochs
        nbatches = len(self.trainer.datamodule.train_dataloader())
        self.total_steps = num_epochs * nbatches
        self.warmup_steps = int(self.hparams.warmup_ratio * self.total_steps)
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.total_steps)

        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step', # or 'epoch'
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, "monitor": "train/loss"}
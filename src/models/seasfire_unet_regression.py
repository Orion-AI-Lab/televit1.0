from typing import Any, List

import torch
from lightning.pytorch import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import AUROC, AveragePrecision, F1Score
import lightning.pytorch as pl
from .components.losses import WeightedMSELoss
import torchmetrics
from torch.nn import ReLU
from transformers import get_cosine_schedule_with_warmup



class plUNET(pl.LightningModule):
    def __init__(
            self,
            input_vars: list = None,
            positional_vars: list = None,
            lr: float = 0.001,
            weight_decay: float = 0.0005,
            loss='mse',
            encoder='efficientnet-b5',
            warmup_ratio=0.05,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = smp.UnetPlusPlus(encoder_name=encoder, in_channels=len(input_vars) + len(positional_vars), classes=1, activation=None)

        assert loss in ['mse', 'weighted_mse'], f"loss should be one of ['mse', 'weighted_mse'], got {loss}"
        if loss == 'mse':
            self.criterion = torch.nn.MSELoss()
        elif loss == 'weighted_mse':
            self.criterion = WeightedMSELoss()

        self.val_mse = torchmetrics.regression.MeanSquaredError()
        self.val_r2 = torchmetrics.regression.R2Score()
        self.test_mse = torchmetrics.regression.MeanSquaredError()
        self.test_r2 = torchmetrics.regression.R2Score()
        self.relu = torch.nn.ReLU()


    def forward(self, x: torch.Tensor):
        y = self.net(x)
        # added relu activation 2025-01-25
        y = self.relu(y)
        return y

    def step(self, batch: Any):
        # TODO remove squeeze once the model is made to handle inputs of shape (c, t, h, w)
        x_local_unsqueezed = batch['x_local']
        x_local = x_local_unsqueezed.squeeze()
        x_local_mask = batch['x_local_mask']
        x_oci = batch['x_oci']
        # TODO remove squeeze once the model is made to handle inputs of shape (c, t, h, w)
        x_global = batch['x_global'].squeeze()
        y_local = batch['y_local']
        y_global = batch['y_global']
        x_local_pos = batch['x_local_pos'].squeeze()
        normalized_weights = batch['normalized_weights'].float().squeeze()

        # if this is the first batch
        if self.global_step == 0:
            # print the shapes of the inputs and outputs
            print(f'x_local shape: {x_local.shape}')
            print(f'y_local shape: {y_local.shape}')
            print(f'x_local_mask shape: {x_local_mask.shape}')
            print(f'x_local_pos shape: {x_local_pos.shape}')


        if self.hparams.positional_vars is not None:
            x_local = torch.cat([x_local, x_local_pos], dim=1)
        x = x_local
        y = y_local

        # calculate pad_size for x_local so that it is divisible by 32
        pad_size = (x_local.shape[2] % 32) // 2
    

        x = x.float()
        # pad x of shape (batch_size, C, 80, 80) to (batch_size, C, 96, 96)
        if pad_size > 0:
            x = torch.nn.functional.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
            # pad weights of shape (batch_size, 80, 80) to (batch_size, 96, 96)
            normalized_weights = torch.nn.functional.pad(normalized_weights, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=1)
            # pad y of shape (batch_size, 80, 80) to (batch_size, 96, 96)
            y = torch.nn.functional.pad(y, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
            if self.global_step == 0:
                print(f'x shape after padding: {x.shape}')
                print(f'normalized_weights shape after padding: {normalized_weights.shape}')

        y = y.float()
        logits = self.forward(x)
        if self.hparams.loss == 'weighted_mse':
            loss = loss = self.criterion(logits.squeeze(), y.squeeze(), weights=normalized_weights.squeeze())
        else:
            loss = self.criterion(logits, y)
        preds = logits.squeeze()
        return loss, preds, y, x_local_unsqueezed

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    # def training_epoch_end(self, outputs: List[Any]):
    #     pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.step(batch)
        # log val metrics
        self.val_mse.update(preds.flatten(), targets.flatten())
        self.val_r2.update(preds.flatten(), targets.flatten())

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds.detach().cpu(), "targets": targets.detach().cpu(),
                "inputs": inputs.detach().cpu()}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.step(batch)
        self.test_mse.update(preds.flatten(), targets.flatten())
        self.test_r2.update(preds.flatten(), targets.flatten())

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True)
        self.log("test/r2", self.test_r2, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds.detach().cpu(), "targets": targets.detach().cpu()}

    # def test_epoch_end(self, outputs: List[Any]):
    #     pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
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

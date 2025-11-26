from typing import Any, List
import torch
from lightning.pytorch import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import AUROC, AveragePrecision, F1Score
import lightning.pytorch as pl
from .components import televit_time_cond
import einops
from transformers import get_cosine_schedule_with_warmup
import numpy as np

class plUNET(pl.LightningModule):
    def __init__(
            self,
            input_vars: list = None, # this is so that the wandb callback can access the input variables
            input_local_shape: list = None,
            input_global_shape: list = None,
            input_oci_shape: list = None,
            patch_local_shape: list = None,
            patch_global_shape: list = None,
            patch_oci_shape: list = None,
            max_lead_time: int = 16,
            min_lead_time: int = 1,
            lr: float = 0.001,
            weight_decay: float = 0.0005,
            loss='ce',
            sea_masked=False,
            pool='mean',
            compile=True,
            dim = 768,
            depth = 12,
            heads = 12,
            mlp_dim = 1536,
            num_register_tokens = 4,
            warmup_ratio=0.05
    ):
        super().__init__()
        self.sea_masked = sea_masked

        self.save_hyperparameters(logger=False)
        
        self.net = televit_time_cond.TeleViT(
            input_dims = [input_local_shape, input_global_shape, input_oci_shape],
            patch_dims = [patch_local_shape, patch_global_shape, patch_oci_shape],
            input_names = ["local", "global", "oci"],
            output_shape_from_input = "local",
            num_classes = 2,
            cond_dim=32,
            num_lead_times=max_lead_time - min_lead_time + 1,
            dim = dim,
            depth = depth,
            heads = heads,
            mlp_dim = mlp_dim,
            num_register_tokens = num_register_tokens,
            pool = pool
        )
        

        # check if torch version is >= 2.0.0
        if torch.__version__ >= "2.0.0" and compile:
            self.net = torch.compile(self.net)
        
        if loss == 'dice':
            self.criterion = smp.losses.DiceLoss(mode='multiclass')
        elif loss == 'ce':
            if self.sea_masked:
                self.criterion = torch.nn.CrossEntropyLoss(ignore_index=2)
            else:
                self.criterion = torch.nn.CrossEntropyLoss()

        if self.sea_masked:
            self.val_auprc = AveragePrecision(num_classes=1,  task="binary", ignore_index=2)
            self.test_auprc = AveragePrecision(num_classes=1, task="binary", ignore_index=2)
        else:
            # create metrics for each lead time
            self.val_auprc = torch.nn.ModuleList([AveragePrecision(task='binary', num_classes=1) for _ in range(max_lead_time - min_lead_time + 1)])
            self.test_auprc = torch.nn.ModuleList([AveragePrecision(task='binary', num_classes=1) for _ in range(max_lead_time - min_lead_time + 1)])
            # self.val_auprc = AveragePrecision(task='binary', num_classes=1)
            # self.test_auprc = AveragePrecision(task='binary', num_classes=1)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x_local = batch['x_local']
        x_local_mask = batch['x_local_mask']
        x_oci = batch['x_oci']
        x_global = batch['x_global']
        y_local = batch['y_local']
        y_global = batch['y_global']
        lead_times = batch['lead_time']
        x = x_local
        y = y_local
        # if this is the first batch
        if self.global_step == 0:
            # print the shapes of the inputs and outputs
            print(f'x_local shape: {x_local.shape}')
            print(f'x_oci shape: {x_oci.shape}')
            print(f'x_global shape: {x_global.shape}')
            print(f'y_local shape: {y_local.shape}')
            print(f'y_global shape: {y_global.shape}')
            print(f'lead_times shape: {lead_times.shape}')

        y = y.long()
        # make x, x_t, x_global into torch.cuda.FloatTensor
        x = x.float()
        x_oci = x_oci.float()
        x_global = x_global.float()
        lead_times = lead_times.long()

        logits = self.net([x,x_global, x_oci], cond_input=lead_times)
        if self.global_step == 0:
            print('logits shape: ', logits.shape)
        if self.sea_masked:
            y[x_local_mask == 1] = 2

        loss = self.criterion(logits, y)
        preds = torch.nn.functional.softmax(logits, dim=1)[:, 1]
        preds[x_local_mask] = 0
        if self.global_step == 0:
            print('preds shape: ', preds.shape)
        return loss, preds, y, x

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.step(batch)

        self.log("train/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True)

        return {"loss": loss}

    # def on_train_epoch_end(self, outputs: List[Any]):
    #     pass

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int):
        loss, preds, targets, inputs = self.step(batch)
        # log val metrics for each lead time

        self.val_auprc[dataloader_idx].update(preds.flatten(), targets.flatten())

        self.log(f"val/auprc[dl_idx={dataloader_idx}]", self.val_auprc[dataloader_idx], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds.detach().cpu(), "targets": targets.detach().cpu(),
                "inputs": inputs.detach().cpu()}

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int):
        loss, preds, targets, _ = self.step(batch)

        self.test_auprc[dataloader_idx].update(preds.flatten(), targets.flatten())
        self.log(f"test/auprc[dl_idx={dataloader_idx}]", self.test_auprc[dataloader_idx], on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds.detach().cpu(), "targets": targets.detach().cpu()}

    def on_train_epoch_end(self):
        # if current epoch is a multiple of 10 and epoch > 0
        if True:
        # if self.current_epoch % 10 == 9:
            self.trainer.datamodule.data_train.update_temperature(increment=0.1, factor=1 - 2 * (self.current_epoch == 50))
            print()
            print(f'Increased temperature to {self.trainer.datamodule.data_train.temperature}')
            print

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

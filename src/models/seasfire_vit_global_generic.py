from typing import Any, List

import torch
from lightning.pytorch import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import AUROC, AveragePrecision, F1Score
import lightning.pytorch as pl
from .components import televit_generic
from transformers import get_cosine_schedule_with_warmup
import einops


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
            warmup_ratio=0.05,
            pos_emb = None,
            patch_emb='linear',
            positional_vars: list = None,
            warmup_epochs=0,
    ):
        super().__init__()
        self.sea_masked = sea_masked
        self.warmup_epochs = warmup_epochs
        self.save_hyperparameters(logger=False)

        assert input_local_shape is not None, "input_local_shape is required"
        assert patch_local_shape is not None, "patch_local_shape is required"

        input_names = ["local", "global", "oci"]
        self.input_names = [input_names[i] for i, x in enumerate([input_local_shape, input_global_shape, input_oci_shape]) if x]
        
        self.pos_emb = pos_emb

        # if the positional variables are provided make sure the input_local_shape is len(input_vars) + len(positional_vars) at the first dimension
        if positional_vars:
            if input_global_shape and patch_global_shape:
                input_global_shape[0] = len(input_vars) + len(positional_vars) # , "input_global_shape[0] should be equal to len(input_vars) + len(positional_vars)"
                patch_global_shape[0] = len(input_vars) + len(positional_vars) # , "patch_global_shape[0] should be equal to len(input_vars) + len(positional_vars)"

            input_local_shape[0] = len(input_vars) + len(positional_vars) # , "input_local_shape[0] should be equal to len(input_vars) + len(positional_vars)"
            patch_local_shape[0] = len(input_vars) + len(positional_vars) # , "patch_local_shape[0] should be equal to len(input_vars) + len(positional_vars)"

        self.net = televit_generic.TeleViT(
            input_dims = [x for x in [input_local_shape, input_global_shape, input_oci_shape] if x],
            patch_dims = [x for x in [patch_local_shape, patch_global_shape, patch_oci_shape] if x],
            input_names = self.input_names,
            output_shape_from_input = "local",
            num_classes = 2,
            dim = dim,
            depth = depth,
            heads = heads,
            mlp_dim = mlp_dim,
            num_register_tokens = num_register_tokens,
            pool = pool,
            pos_emb = pos_emb,
            patch_emb=patch_emb
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
            self.val_auprc = AveragePrecision(task='binary', num_classes=1)
            self.test_auprc = AveragePrecision(task='binary', num_classes=1)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x_local = batch['x_local']
        x_local_mask = batch['x_local_mask']
        x_oci = batch['x_oci']
        x_global = batch['x_global']
        y_local = batch['y_local']
        y_global = batch['y_global']

        if self.pos_emb == 'satclip':
            x_local_satclip = batch['x_local_satclip'].float()
            x_global_satclip = batch['x_global_satclip'].float()

        x_local_pos = batch['x_local_pos'].float()
        x_global_pos = batch['x_global_pos'].float()
        x = x_local
        y = y_local

        y = y.long()
        x = x.float()
        x_oci = x_oci.float()
        x_global = x_global.float()


        if len(self.input_names) == 1 and self.warmup_epochs > 0:
            raise ValueError("warmup_epochs is not supported for single input models")
        if len(self.input_names) == 2 and self.current_epoch < self.warmup_epochs:
            x = torch.zeros_like(x)

        if len(self.input_names) == 3 and self.current_epoch < self.warmup_epochs:
            x = torch.zeros_like(x)
            # first half of the warmup, set x_global to 0
            if 'global' in self.input_names and self.current_epoch < (self.warmup_epochs // 2):
                x_global = torch.zeros_like(x_global)

        if self.hparams.positional_vars:
            # x_local_pos (b, c, h, w) -> (b, c, t, h, w)
            x_local_pos = einops.repeat(x_local_pos, 'b c h w -> b c t h w', t=x_local.shape[2])

            # x_global_pos (b, c, h, w) -> (b, c, t, h, w)
            x_global_pos = einops.repeat(x_global_pos, 'b c h w -> b c t h w', t=x_global.shape[2])


            #  stack x and x_local_pos
            x = torch.cat([x, x_local_pos], dim=1)

            #  stack x_global and x_global_pos
            x_global = torch.cat([x_global, x_global_pos], dim=1)

            if self.global_step == 0:
                print(f'x shape after stacking with x_local_pos: {x.shape}')
                print(f'x_global shape after stacking with x_global_pos: {x_global.shape}')


        input_dict = {'local': x, 'oci': x_oci, 'global': x_global}

        # if this is the first batch
        if self.global_step == 0:
            # print the shapes of the inputs and outputs
            print(f'x_local shape: {x_local.shape}')
            print(f'x_oci shape: {x_oci.shape}')
            print(f'x_global shape: {x_global.shape}')
            print(f'y_local shape: {y_local.shape}')
            print(f'y_global shape: {y_global.shape}')
            if self.pos_emb == 'satclip':
                print(f'x_local_satclip shape: {x_local_satclip.shape}')
                print(f'x_global_satclip shape: {x_global_satclip.shape}')
            print(f'x_local_pos shape: {x_local_pos.shape}')
            print(f'x_global_pos shape: {x_global_pos.shape}')  
            print('input_dict size:', {k: v.shape for k, v in input_dict.items()})


        if self.pos_emb == 'satclip':
            logits = self.net([input_dict[x] for x in self.input_names], x_local_satclip, x_global_satclip)
        else:
            logits = self.net([input_dict[x] for x in self.input_names])


        if self.global_step == 0:
            print('logits shape: ', logits.shape)
        if self.sea_masked:
            y[x_local_mask == 1] = 2

        loss = self.criterion(logits, y)
        preds = torch.nn.functional.softmax(logits, dim=1)[:, 1]
        # preds[x_local_mask] = 0
        if self.global_step == 0:
            print('preds shape: ', preds.shape)
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

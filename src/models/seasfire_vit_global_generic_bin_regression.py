from typing import Any, List

import torch
from lightning.pytorch import LightningModule
import torchmetrics
from torchmetrics import MaxMetric
import torchmetrics.classification
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import AUROC, AveragePrecision, F1Score
import segmentation_models_pytorch as smp
import lightning.pytorch as pl
import torchmetrics.regression
from .components import televit_generic
from .components.losses import WeightedMSELoss
from transformers import get_cosine_schedule_with_warmup
import einops
from torch.nn import functional as F
from .components.bin_regression import cls_head, get_log_bins, get_bin_centers, soft_bin_to_continuous

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
            pos_emb = 'learnable',
            patch_emb='linear',
            positional_vars: list = None,
            latent_classes: int = 24,
            num_bins: int = 24,
            bin_spacing : str = 'log'
    ):
        super().__init__()
        self.sea_masked = sea_masked

        self.save_hyperparameters(logger=False)

        if sea_masked and loss == 'weighted_mse':
            raise NotImplementedError("Weighted MSE loss is not implemented for sea_masked=True")
        
        input_names = ["local", "global", "oci"]
        self.input_names = [input_names[i] for i, x in enumerate([input_local_shape, input_global_shape, input_oci_shape]) if x]
        
        self.pos_emb = pos_emb

        # if the positional variables are provided make sure the input_local_shape is len(input_vars) + len(positional_vars) at the first dimension
        if positional_vars:
            input_local_shape[0] = len(input_vars) + len(positional_vars) # , "input_local_shape[0] should be equal to len(input_vars) + len(positional_vars)"
            input_global_shape[0] = len(input_vars) + len(positional_vars) # , "input_global_shape[0] should be equal to len(input_vars) + len(positional_vars)"
            patch_local_shape[0] = len(input_vars) + len(positional_vars) # , "patch_local_shape[0] should be equal to len(input_vars) + len(positional_vars)"
            patch_global_shape[0] = len(input_vars) + len(positional_vars) # , "patch_global_shape[0] should be equal to len(input_vars) + len(positional_vars)"
        
        
        self.net = televit_generic.TeleViT(
            input_dims = [x for x in [input_local_shape, input_global_shape, input_oci_shape] if x],
            patch_dims = [x for x in [patch_local_shape, patch_global_shape, patch_oci_shape] if x],
            input_names = self.input_names,
            output_shape_from_input = "local",
            num_classes = latent_classes,
            dim = dim,
            depth = depth,
            heads = heads,
            mlp_dim = mlp_dim,
            num_register_tokens = num_register_tokens,
            pool = pool,
            pos_emb=pos_emb,
            patch_emb=patch_emb
        )

        # check if torch version is >= 2.0.0
        if torch.__version__ >= "2.0.0" and compile:
            self.net = torch.compile(self.net)
        
        if loss == 'dice':
            self.criterion = smp.losses.DiceLoss(mode='multiclass')
        elif loss == 'ce':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("loss should be either 'dice' or 'ce'")

        self.val_mse = torchmetrics.regression.MeanSquaredError()
        self.val_r2 = torchmetrics.regression.R2Score()
        self.test_mse = torchmetrics.regression.MeanSquaredError()
        self.test_r2 = torchmetrics.regression.R2Score()
        self.relu = torch.nn.ReLU()
        self.num_bins = num_bins
        if latent_classes != num_bins:
            self.cls_head = cls_head(in_channels=latent_classes, bins=num_bins).to(self.device)
        else:
            self.cls_head = None
        if bin_spacing == 'log':
            self.bins = get_log_bins(num_bins, 0.00001, 1.0).to(self.device)
        elif bin_spacing == 'linear':
            self.bins = torch.linspace(0.00001, 1.0, num_bins).to(self.device)
        else:
            raise ValueError("bin_spacing should be either 'log' or 'linear'")
        # add bin for practically zero
        self.bins = torch.cat([torch.tensor([0.0]), self.bins]).to(self.device)
        print("Bins: ", self.bins)


    def forward(self, x: torch.Tensor):
        y = self.net(x)
        return y


    def step(self, batch: Any):
        x_local = batch['x_local']
        x_local_mask = batch['x_local_mask']
        x_oci = batch['x_oci']
        x_global = batch['x_global']
        y_local = batch['y_local'].float()
        y_global = batch['y_global']
        x_local_satclip = batch['x_local_satclip'].float()
        x_global_satclip = batch['x_global_satclip'].float()
        x_local_pos = batch['x_local_pos']
        x_global_pos = batch['x_global_pos']
        normalized_weights = batch['normalized_weights'].float()

        x = x_local
        y = y_local

        y_bins = torch.bucketize(y_local, self.bins.to(y_local.device)) - 1  # Get bin indices
        y_bins = torch.clamp(y_bins, min=0, max=self.num_bins - 1).long() 

        # if this is the first batch
        if self.global_step == 0:
            # print the shapes of the inputs and outputs
            print(f'x_local shape: {x_local.shape}')
            print(f'x_oci shape: {x_oci.shape}')
            print(f'x_global shape: {x_global.shape}')
            print(f'y_local shape: {y_local.shape}')
            print(f'y_global shape: {y_global.shape}')
            print(f'x_local_satclip shape: {x_local_satclip.shape}')
            print(f'x_global_satclip shape: {x_global_satclip.shape}')
            print(f'x_local_pos shape: {x_local_pos.shape}')
            print(f'x_global_pos shape: {x_global_pos.shape}')
            print(f'normalized_weights shape: {normalized_weights.shape}')
            print(f'y_bins shape: {y_bins.shape}')
            
            if self.sea_masked:
                print("Sea is masked")
            else:
                print("Sea is not masked")

        x = x.float()
        x_oci = x_oci.float()
        x_global = x_global.float()

        x_local_pos = x_local_pos.float()
        x_global_pos = x_global_pos.float()

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

        if self.pos_emb == 'satclip':
            logits = self.net([input_dict[x] for x in self.input_names], x_local_satclip, x_global_satclip)
        else:
            logits = self.net([input_dict[x] for x in self.input_names])

        if self.cls_head:
            logits = self.cls_head(logits)

        if self.global_step == 0:
            print('logits shape: ', logits.shape)

        loss = self.criterion(logits.squeeze(), y_bins.squeeze())

        preds = soft_bin_to_continuous(logits, self.bins) 

        if self.global_step == 0:
            print('logits shape: ', logits.shape)
            print('preds shape: ', preds.shape)

        return loss, preds, y, x, x_local_mask.flatten(), y_bins

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs, x_local_mask, y_bins = self.step(batch)

        self.log("train/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True)

        return {"loss": loss}        


    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs, x_local_mask, y_bins = self.step(batch)
        # log val metrics
        self.val_mse.update(preds.flatten()[x_local_mask==0], targets.flatten()[x_local_mask==0])
        self.val_r2.update(preds.flatten()[x_local_mask==0], targets.flatten()[x_local_mask==0])

        
        # y_bins = einops.rearrange(y_bins, "b c h w -> b c (h w)")
        # targets = einops.rearrange(targets, "b h w -> b (h w)" )
        # self.val_auprc.update(y_bins, targets)
        # self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds.detach().cpu(), "targets": targets.detach().cpu(),
                "inputs": inputs.detach().cpu()}


    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _, x_local_mask, y_bins = self.step(batch)

        self.test_mse.update(preds.flatten()[x_local_mask==0], targets.flatten()[x_local_mask==0])
        self.test_r2.update(preds.flatten()[x_local_mask==0], targets.flatten()[x_local_mask==0])

        # y_bins = einops.rearrange(y_bins, "b c h w -> b c (h w)")
        # targets = einops.rearrange(targets, "b h w -> b (h w)" )
        # self.val_auprc.update(y_bins, targets)
        # self.test_auprc.update(y_bins.flatten(), targets.flatten())
        # self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/r2", self.test_r2, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds.detach().cpu(), "targets": targets.detach().cpu()}


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

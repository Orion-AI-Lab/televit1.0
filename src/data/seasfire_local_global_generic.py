from typing import Optional, Tuple
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import numpy as np
import xarray as xr
import json
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from .components.seasfire_dataset_generic import create_dataset_for_years, get_loaded_datasets
import os
from pathlib import Path
import time

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class SeasFireLocalGlobalDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self,
            ds_path: str = None,
            ds_path_global: str = None,
            input_vars: list = None,
            positional_vars: list = None,
            oci_vars: list = None,
            local_lag: int = 1,
            global_lag: int = 1,
            oci_lag: int = 10,
            log_transform_vars: list = None,
            target: str = 'gwis_ba',
            task: str = 'classification',
            target_shift: int = 1,
            input_local_shape: list = None,
            batch_size: int = 64,
            num_workers: int = 8,
            pin_memory: bool = False,
            debug: bool = False,
            stats_dir: str = os.getcwd() + '/stats',
            patch_local_shape = None,
            patch_global_shape = None,
            posemb_path = None,
            posemb_global_path = None,
            posemb_var_name = None,
            posemb_mask='lsm',
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        if input_local_shape is None:
            input_local_shape = [1, 80, 80]
        if positional_vars is None:
            self.positional_vars = []
        else:
            self.positional_vars = positional_vars
        self.log_transform_vars = log_transform_vars
        self.save_hyperparameters(logger=False)
        self.ds_path = ds_path
        self.ds_path_global = ds_path_global
        self.input_vars = list(input_vars)
        self.oci_vars = list(oci_vars)
        self.oci_lag = oci_lag
        self.target = target
        self.target_shift = target_shift
        self.ds = xr.open_zarr(ds_path, consolidated=True)
        self.mean_std_dict = None
        self.debug = debug
        if self.debug:
            self.training_years = [2002, 2003]
            self.validation_years = [2002, 2003]
            self.test_years = [2002, 2003]
            self.selected_years = [2002, 2003]
        else:
            self.training_years = list(range(2003, 2018))
            self.validation_years = [2018]
            self.test_years = [2019]
            self.selected_years = None
        self.input_local_shape = tuple(input_local_shape)
        self.task = task
        self.local_lag = local_lag
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.stats_dir = stats_dir
        self.global_lag = global_lag
        self.patch_local_shape = patch_local_shape
        self.patch_global_shape = patch_global_shape
        self.posemb_path = posemb_path
        self.posemb_global_path = posemb_global_path
        self.posemb_var_name = posemb_var_name
        self.posemb_mask = posemb_mask

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            print(self.ds[self.input_vars])


            # calculate time for loading datasets
            start = time.time()
            print('Loading datasets...')
            ds, global_ds, oci_ds = get_loaded_datasets(self.ds_path, self.ds_path_global, self.input_vars, self.log_transform_vars, self.oci_vars, keep_vars=['area', 'log_gwis_ba_anom','log_gwis_ba',  'gwis_ba', 'gfed_region', 'ndvi', 'pop_dens', 'lsm', 'burnable_fraction', 'gwis_ba_frac'], selected_years=self.selected_years)
            end = time.time()
            print(f'Datasets loaded in {end - start:.2f} seconds')

            print('Creating training dataset...')
            self.data_train = create_dataset_for_years(ds, global_ds, oci_ds, self.input_vars, self.oci_vars, self.positional_vars, 
                                                       self.target,min_lead_time=self.target_shift, max_lead_time=self.target_shift, local_lag=self.local_lag, 
                                                       oci_lag=self.oci_lag, global_lag=self.global_lag, patch_size=self.input_local_shape, years=self.training_years,
                                                       task=self.task, patch_local_shape=self.patch_local_shape, patch_global_shape=self.patch_global_shape,
                                                       posemb_path=self.posemb_path, posemb_global_path=self.posemb_global_path, posemb_var_name=self.posemb_var_name, posemb_mask=self.posemb_mask)
            
            print('Creating validation dataset...')
            self.data_val = create_dataset_for_years(ds, global_ds, oci_ds, self.input_vars, self.oci_vars, self.positional_vars, 
                                            self.target,min_lead_time=self.target_shift, max_lead_time=self.target_shift, local_lag=self.local_lag, 
                                            oci_lag=self.oci_lag, global_lag=self.global_lag, patch_size=self.input_local_shape, years=self.validation_years, 
                                            task=self.task, patch_local_shape=self.patch_local_shape, patch_global_shape=self.patch_global_shape,
                                            posemb_path=self.posemb_path, posemb_global_path=self.posemb_global_path, posemb_var_name=self.posemb_var_name, posemb_mask=self.posemb_mask)

            # shuffle the validation dataset
            from torch.utils.data import Subset
            val_size = len(self.data_val)
            val_indices = list(range(val_size))
            np.random.shuffle(val_indices)
            val_indices = val_indices[:val_size]
            self.data_val = Subset(self.data_val, val_indices)

            
            print('Creating test dataset...')
            self.data_test = create_dataset_for_years(ds, global_ds, oci_ds, self.input_vars, self.oci_vars, self.positional_vars, 
                                            self.target,min_lead_time=self.target_shift, max_lead_time=self.target_shift, local_lag=self.local_lag, 
                                            oci_lag=self.oci_lag, global_lag=self.global_lag, patch_size=self.input_local_shape, years=self.test_years, 
                                            task=self.task, patch_local_shape=self.patch_local_shape, patch_global_shape=self.patch_global_shape,
                                            posemb_path=self.posemb_path, posemb_global_path=self.posemb_global_path, posemb_var_name=self.posemb_var_name, posemb_mask=self.posemb_mask)
                                            


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            # persistent_workers=True
        )

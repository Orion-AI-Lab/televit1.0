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
            target: str = 'BurntArea',
            min_lead_time: int = 1,
            max_lead_time: int = 16,
            local_input_shape: list = None,
            batch_size: int = 64,
            num_workers: int = 8,
            pin_memory: bool = False,
            debug: bool = False,
            stats_dir: str = os.getcwd() + '/stats',
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        if local_input_shape is None:
            local_input_shape = [1, 80, 80]
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
        self.min_lead_time = min_lead_time
        self.max_lead_time = max_lead_time
        self.ds = xr.open_zarr(ds_path, consolidated=True)
        # TODO remove when we have the new datacube
        self.ds['sst'] = self.ds['sst'].where(self.ds['sst'] >= 0)
        self.mean_std_dict = None
        self.debug = debug
        self.training_years = list(range(2002, 2018))
        self.validation_years = [2018]
        self.test_years = [2019]
        self.local_input_shape = tuple(local_input_shape)
        self.local_lag = local_lag
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.stats_dir = stats_dir
        self.global_lag = global_lag

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            print(self.ds[self.input_vars])


            print('Loading datasets...')
            ds, global_ds, oci_ds = get_loaded_datasets(self.ds_path, self.ds_path_global, self.input_vars, self.log_transform_vars, self.oci_vars, keep_vars=['gwis_ba', 'gfed_region', 'ndvi', 'pop_dens'])
            print('Datasets loaded...')

            print('Creating training dataset...')
            self.data_train = create_dataset_for_years(ds, global_ds, oci_ds, self.input_vars, self.oci_vars, self.positional_vars, 
                                                       self.target,min_lead_time=self.min_lead_time, max_lead_time=self.max_lead_time, local_lag=self.local_lag, 
                                                       oci_lag=self.oci_lag, global_lag=self.global_lag, patch_size=self.local_input_shape, years=self.training_years)
            
            print('Creating validation/testing datasets...')
            # create a list of validation/testing datasets for each lead_time
            self.data_val = []
            self.data_test = []
            for lead_time in range(self.min_lead_time, self.max_lead_time + 1):
                self.data_val.append(create_dataset_for_years(ds, global_ds, oci_ds, self.input_vars, self.oci_vars, self.positional_vars, 
                                                self.target,min_lead_time=lead_time, max_lead_time=lead_time, local_lag=self.local_lag, 
                                                oci_lag=self.oci_lag, global_lag=self.global_lag, patch_size=self.local_input_shape, years=self.validation_years))
    
                self.data_test.append(create_dataset_for_years(ds, global_ds, oci_ds, self.input_vars, self.oci_vars, self.positional_vars, 
                                                self.target,min_lead_time=lead_time, max_lead_time=lead_time, local_lag=self.local_lag, 
                                                oci_lag=self.oci_lag, global_lag=self.global_lag, patch_size=self.local_input_shape, years=self.test_years))
                

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=False
        )

    def val_dataloader(self):
        return [DataLoader(
            dataset=x,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True
        ) for x in self.data_val]

    def test_dataloader(self):
        return [DataLoader(
            dataset=x,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True
        ) for x in self.data_test]

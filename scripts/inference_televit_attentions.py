import xarray as xr
import zarr
import numpy as np
import gc
from tqdm import tqdm
from pathlib import Path
import sys
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import xarray as xr
from tqdm import tqdm
import numpy as np
import pandas as pd
# save dictionary to json file in path
import json
import dotenv
import os 
import time
import argparse
import datetime

dotenv.load_dotenv(dotenv.find_dotenv())
sys.path.insert(0, './')

from src.models.seasfire_vit_global_generic import plUNET
from functools import wraps
import torch
from torch import nn
from src.models.components.televit_generic import Attention


def rollout_attention_per_head_with_residual(attentions):
    """
    Calculate attention rollout per head with residual connections by cumulative matrix multiplication across layers.
    
    Parameters:
    attentions (torch.Tensor): A tensor of shape (batch_size, num_layers, num_heads, num_tokens, num_tokens)
    
    Returns:
    torch.Tensor: Rollout attention for each head, shape (batch_size, num_heads, num_tokens, num_tokens)
    """
    # Get the shape of the input tensor
    batch_size, num_layers, num_heads, num_tokens, _ = attentions.shape
    # Start with the identity matrix for each head in each batch
    rollout_attention = torch.eye(num_tokens).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_heads, 1, 1).to(attentions.device)
    # Cumulatively multiply each layer's attention matrix onto the rollout attention, with residuals
    for layer in range(num_layers):
        # Add identity matrix to simulate residual connection
        layer_attention_with_residual = attentions[:, layer, :, :, :]/2 + torch.eye(num_tokens).to(attentions.device)/2
        # Normalize each layer’s attention matrix with residuals to maintain probabilistic interpretation
        layer_attention_with_residual /= layer_attention_with_residual.sum(dim=-1, keepdim=True)
        # Update rollout attention
        rollout_attention = rollout_attention @ layer_attention_with_residual
    
    return rollout_attention


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

class Recorder(nn.Module):
    def __init__(self, vit, device = None):
        super().__init__()
        self.vit = vit

        self.data = None
        self.recordings = []
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device

    def _hook(self, _, input, output):
        self.recordings.append(output.clone().detach())

    def _register_hook(self):
        modules = find_modules(self.vit.transformer, Attention)
        for module in modules:
            handle = module.attend.register_forward_hook(self._hook)
            self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.vit

    def clear(self):
        self.recordings.clear()

    def record(self, attn):
        recording = attn.clone().detach()
        self.recordings.append(recording)

    def forward(self, inputs, *args, **kwargs):
        assert not self.ejected, 'recorder has been ejected, cannot be used anymore'
        self.clear()
        if not self.hook_registered:
            self._register_hook()

        pred = self.vit(inputs, *args, **kwargs)

        # move all recordings to one device before stacking
        target_device = self.device if self.device is not None else inputs[0].device
        recordings = tuple(map(lambda t: t.to(target_device), self.recordings))

        attns = torch.stack(recordings, dim = 1) if len(recordings) > 0 else None
        return pred, attns



# split x_local in 80x80 patches
def split_x_local(x_local, patch_size=80):
    x_local_patches = []
    for i in range(0, x_local.shape[1], patch_size):
        for j in range(0, x_local.shape[2], patch_size):
            x_local_patches.append(x_local[:, i:i+patch_size, j:j+patch_size])
    return x_local_patches

def split_x_local_emb(x_local, patch_size=80):
    x_local_patches = []
    for i in range(0, x_local.shape[0], patch_size):
        for j in range(0, x_local.shape[1], patch_size):
            x_local_patches.append(x_local[i:i+patch_size, j:j+patch_size, :])
    return x_local_patches


# unite x_local patches
def unite_x_local(x_local_patches, x_local_shape, patch_size=80):
    x_local = np.zeros(x_local_shape)
    for i in range(0, x_local.shape[1], patch_size):
        for j in range(0, x_local.shape[2], patch_size):
            x_local[:, i:i+patch_size, j:j+patch_size] = x_local_patches.pop(0)
    return x_local

def load_day_ds(ds, ds_global, oci_ds, day, input_vars,positional_vars, oci_vars, oci_lag, target, target_shift):
    ds_day = ds.isel(time=day)
    oci_ds =  oci_ds.sel(time=slice(ds_day.time - np.timedelta64(oci_lag * 31, 'D'), ds_day.time))
    oci_ds = oci_ds.isel(time=slice(-oci_lag, None))
    ds_day_global = ds_global.isel(time=day)
    # calculate target variable
    target = ds.isel(time=day+target_shift)[target]
    x_local = np.stack([ds_day[var].values for var in input_vars + positional_vars], axis=0)
    x_local = np.nan_to_num(x_local, nan=-1)
    x_global = np.stack([ds_day_global[var].values for var in input_vars + positional_vars], axis=0)
    x_global = np.nan_to_num(x_global, nan=-1)
    x_t = np.stack([oci_ds[var].values for var in oci_vars], axis=0) 
    x_t = np.nan_to_num(x_t, nan=0)
    return x_local, x_global, x_t, target


def predict_day_televit(ds, ds_global, oci_ds, day, target_shift, model, oci_vars, ds_coarse=None):
    day = day - target_shift

    x_local, x_global, x_t, target = load_day_ds(ds, ds_global, oci_ds, day, model.hparams['input_vars'], model.hparams['positional_vars'],
                                                  oci_vars, 10, 'gwis_ba', 0 )
    x_local_patches = split_x_local(x_local, 80)
    local_embeddings = split_x_local_emb(local_embedding_all, 5)
    y_local_patches = []

    input_local = torch.from_numpy(np.stack(x_local_patches)).unsqueeze(2).float().to('cuda')
    # new_global_shape = (input_local.shape[0], input_local.shape[1], input_local.shape[2], input_local.shape[3], input_local.shape[4])
    input_global = torch.from_numpy(x_global).expand(input_local.shape[0], -1, -1,-1).unsqueeze(2).float().to('cuda')
    input_t = torch.from_numpy(x_t).expand(input_local.shape[0], -1, -1).float().to('cuda')
    # print(input_local.shape, input_global.shape, input_t.shape)
    batch_size = 16  # Define your batch size
    num_batches = (input_local.shape[0] + batch_size - 1) // batch_size  # Calculate the number of batches

    attns_list = []
    preds_list = []
    net = Recorder(model.net, device='cuda')

    with torch.no_grad():

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, input_local.shape[0])

            input_local_batch = input_local[start_idx:end_idx]
            input_global_batch = input_global[start_idx:end_idx]
            input_t_batch = input_t[start_idx:end_idx]
            preds_batch, attns_batch = net([input_local_batch, input_global_batch, input_t_batch])
            attns_batch = rollout_attention_per_head_with_residual(attns_batch)
            attns_list.append(attns_batch.detach().cpu().numpy())
            
            preds_batch = F.softmax(preds_batch, dim=1).squeeze()[:, 1]
            for i in range(preds_batch.shape[0]):
                preds_list.append(preds_batch[i].squeeze().cpu().numpy())

   
    # apply softmax
    net.clear()
    net.eject()
    # concatenate all attns
    attns = np.concatenate(attns_list, axis=0)
    # calculate mean over heads
    attns = attns.mean(axis=1)
    
    preds = unite_x_local(preds_list, (1, 720, 1440))[0]
    preds[ds.lsm==0] = np.nan

    num_tokens = model.net.num_patches + model.net.num_register_tokens

    preds_ds = xr.DataArray(preds, dims=['latitude', 'longitude'], coords={'latitude': ds.latitude, 'longitude': ds.longitude}).load()


    attns_ds = xr.DataArray(attns.reshape(9, 18, num_tokens, num_tokens), 
                            dims=['latitude', 'longitude',  'token_x', 'token_y'], 
                            coords={'longitude': ds_coarse['longitude'],
                                    'latitude': ds_coarse['latitude'], 
                                    'token_x': np.arange(attns.shape[-1]), 
                                    'token_y': np.arange(attns.shape[-1])}).load()

    return preds_ds, attns_ds

def load_datasets(ds_path, global_ds_path, input_vars, log_transform_vars, oci_vars, keep_vars=['log_gwis_ba', 'gwis_ba', 'gfed_region', 'ndvi', 'pop_dens', 'lsm'], selected_years=None, stats_dict=None):
    ds = xr.open_zarr(ds_path)
    global_ds = xr.open_zarr(global_ds_path)

    if selected_years is not None:
        ds = ds.sel(time=ds.time.dt.year.isin(selected_years))
        global_ds = global_ds.sel(time=global_ds.time.dt.year.isin(selected_years))

    for var in log_transform_vars:
        ds[var] = np.log(ds[var] + 1)
        global_ds[var] = np.log(global_ds[var] + 1)

    # function to add positional embedding to dataset
    def _add_positional_vars(ds):
        # compute positional embedding from longitude and latitude
        lon = ds.longitude.values
        lat = ds.latitude.values
        lon = np.expand_dims(lon, axis=0)
        lat = np.expand_dims(lat, axis=1)
        lon = np.tile(lon, (lat.shape[0], 1))
        lat = np.tile(lat, (1, lon.shape[1]))
        ds['cos_lon'] = ({'latitude': ds.latitude, 'longitude': ds.longitude}, np.cos(lon * np.pi / 180))
        ds['cos_lat'] = ({'latitude': ds.latitude, 'longitude': ds.longitude}, np.cos(lat * np.pi / 180))
        ds['sin_lon'] = ({'latitude': ds.latitude, 'longitude': ds.longitude}, np.sin(lon * np.pi / 180))
        ds['sin_lat'] = ({'latitude': ds.latitude, 'longitude': ds.longitude}, np.sin(lat * np.pi / 180))
        return ds

    # normalize input variables
    for var in input_vars:
        ds[var] = (ds[var] - stats_dict['local'][var + '_mean']) / stats_dict['local'][var + '_std']
        global_ds[var] = (global_ds[var] - stats_dict['global'][var + '_mean']) / stats_dict['global'][var + '_std']

    ds['log_gwis_ba'] = np.log(ds['gwis_ba'] + 1)
    global_ds['log_gwis_ba'] = np.log(global_ds['gwis_ba'] + 1)

    # concatenate all vars to keep
    vars_to_keep = keep_vars + input_vars + oci_vars
    # remove duplicates and vars that are not in ds
    vars_to_keep = [x for x in list(set(vars_to_keep)) if x in ds.data_vars]

    # load into memory
    ds = ds[vars_to_keep].load()
    global_ds = global_ds[vars_to_keep].load()

    oci_ds = xr.Dataset()
    for var in oci_vars:
        # resample var to 1 month
        oci_ds[var] = ds[var].fillna(0).resample(time='1M').mean(dim='time')
        # normalize oci variables
        # oci_ds[var] = oci_ds[var] / oci_ds[var].std()
        oci_ds[var] = (oci_ds[var] - stats_dict['oci'][var + '_mean']) / stats_dict['oci'][var + '_std']

    oci_ds.load()

    ds = _add_positional_vars(ds)
    global_ds = _add_positional_vars(global_ds)    

    return ds, global_ds, oci_ds

if __name__ == '__main__':     
    # load datacubes
    ds_path = os.environ['DATASET_PATH']
    ds_1deg_path = os.environ['DATASET_PATH_GLOBAL']
    STATS_DIR = './stats'

    print(ds_path)
    print(ds_1deg_path)


    stats_dict = {}

    for dataset in ['local', 'global', 'oci']:
        stats_dict[dataset] = xr.open_dataset(f'{STATS_DIR}/{var}_mean_std.nc')

    def get_model_path(dirpath):
        #  list the dirpath and find the latest checkpoint with this format epoch_{epoch:03d}.ckpt
        checkpoints = [f for f in os.listdir(dirpath) if f.startswith('epoch_') and f.endswith('.ckpt')]
        checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        return os.path.join(dirpath, checkpoints[-1])

    parser = argparse.ArgumentParser(description='Inference script for Televit model.')
    parser.add_argument('--model_checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    args = parser.parse_args()


    model_checkpoint_path = args.model_checkpoint_path
    model = plUNET.load_from_checkpoint(model_checkpoint_path).to('cuda')
    target_shift = model.hparams['target_shift']

    start = time.time()
    print('Loading datasets...')
    # hard coded oci variables
    oci_vars = [
    'oci_censo',
    'oci_ea',
    'oci_epo',
    'oci_ao',
    'oci_nao',
    'oci_nina34_anom',
    'oci_pdo',
    'oci_pna',
    'oci_soi',
    'oci_wp'
    ]

    ds, ds_global, oci_ds = load_datasets(ds_path, ds_1deg_path, log_transform_vars=['tp', 'pop_dens'], input_vars=model_dict[16].hparams['input_vars'], oci_vars=oci_vars, selected_years=None, stats_dict=stats_dict)
    print('Time to load datasets:', time.time() - start)

    dataset = xr.Dataset()


    ds_coarse = ds.coarsen(latitude=80, longitude=80).mean()

    ds_2019 = ds.sel(time=slice('2019-01-01', '2019-12-31'))
    # get the index of the first day of 2019 in ds
    first_day_2019 = ds['time'].to_index().get_loc(ds_2019['time'][0].values)
    print("First day of 2019", first_day_2019)

    num_tokens = model.net.num_patches + model.net.num_register_tokens
    ds_2019[f'predictions'] = xr.DataArray(np.zeros_like(ds_2019.gwis_ba), 
        dims=('time', 'latitude', 'longitude'), 
        coords={'time': ds_2019['time'],
        'latitude': ds_2019['latitude'], 
        'longitude': ds_2019['longitude']})
    dataset[f'attentions'] = xr.DataArray(np.zeros(shape=(len(ds_2019['time']), 18, 9, num_tokens, num_tokens)), 
                                            dims=('time', 'longitude', 'latitude', 'token_x', 'token_y'), 
                                            coords={'time': ds_2019['time'],
                                                'longitude': ds_coarse['longitude'], 
                                                'latitude': ds_coarse['latitude'],
                                                'token_x': np.arange(num_tokens),
                                                'token_y': np.arange(num_tokens)
                                                }
                                                )


    for i, time in tqdm(enumerate(ds_2019['time']), total=len(ds_2019['time'])):
        
        ds_2019[f'predictions'].loc[dict(time=time)], dataset[f'attentions'].loc[dict(time=time)] = predict_day_televit(ds, ds_global, oci_ds, i + first_day_2019, target_shift, model, oci_vars, ds_coarse)
        torch.cuda.empty_cache()

    # save to zarr
    ds_attentions = dataset.attentions
    ds_attentions.load()

    ds_predictions = ds_2019.predictions
    ds_predictions.load()
    
    # Generate unique suffix based on timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    # Create output file paths with unique suffix using Path
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    attentions_path = output_path / f'atts_2019_{target_shift}_{timestamp}.zip'
    predictions_path = output_path / f'preds_2019_{target_shift}_{timestamp}.zip'
    
    print("Writing to zarr store")
    print(f"Attentions output: {attentions_path}")
    print(f"Predictions output: {predictions_path}")
    
    with zarr.ZipStore(attentions_path, mode='w') as store: 
        ds_attentions.chunk(time=4, longitude=-1, latitude=-1, token_x=-1, token_y=-1).to_zarr(store)

    with zarr.ZipStore(predictions_path, mode='w') as store:
        ds_predictions.chunk(time=4, longitude=-1, latitude=-1).to_zarr(store)

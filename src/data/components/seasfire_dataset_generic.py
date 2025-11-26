import xbatcher
import xarray as xr
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from torchvision import transforms
import random
import math
import torch

# save dictionary to json file in path
import json

def normalized_latitude_weights(data: xr.DataArray) -> xr.DataArray:
    """
    Adapted from from https://github.com/google-deepmind/graphcast

    Weights based on latitude, roughly proportional to grid cell area.

    This method supports two use cases only (both for equispaced values):
    * Latitude values such that the closest value to the pole is at latitude
      (90 - d_lat/2), where d_lat is the difference between contiguous latitudes.
      For example: [-89, -87, -85, ..., 85, 87, 89]) (d_lat = 2)
      In this case each point with `lat` value represents a sphere slice between
      `lat - d_lat/2` and `lat + d_lat/2`, and the area of this slice would be
      proportional to:
      `sin(lat + d_lat/2) - sin(lat - d_lat/2) = 2 * sin(d_lat/2) * cos(lat)`, and
      we can simply omit the term `2 * sin(d_lat/2)` which is just a constant
      that cancels during normalization.
    * Latitude values that fall exactly at the poles.
      For example: [-90, -88, -86, ..., 86, 88, 90]) (d_lat = 2)
      In this case each point with `lat` value also represents
      a sphere slice between `lat - d_lat/2` and `lat + d_lat/2`,
      except for the points at the poles, that represent a slice between
      `90 - d_lat/2` and `90` or, `-90` and  `-90 + d_lat/2`.
      The areas of the first type of point are still proportional to:
      * sin(lat + d_lat/2) - sin(lat - d_lat/2) = 2 * sin(d_lat/2) * cos(lat)
      but for the points at the poles now is:
      * sin(90) - sin(90 - d_lat/2) = 2 * sin(d_lat/4) ^ 2
      and we will be using these weights, depending on whether we are looking at
      pole cells, or non-pole cells (omitting the common factor of 2 which will be
      absorbed by the normalization).

      It can be shown via a limit, or simple geometry, that in the small angles
      regime, the proportion of area per pole-point is equal to 1/8th
      the proportion of area covered by each of the nearest non-pole point, and we
      test for this in the test.

    Args:
      data: `DataArray` with latitude coordinates.
    Returns:
      Unit mean latitude weights.
    """
    latitude = data.coords['latitude']

    if np.any(np.isclose(np.abs(latitude), 90.)):
        weights = _weight_for_latitude_vector_with_poles(latitude)
    else:
        weights = _weight_for_latitude_vector_without_poles(latitude)

    return weights / weights.mean(skipna=False)


def _weight_for_latitude_vector_without_poles(latitude):
    """Weights for uniform latitudes of the form [+-90-+d/2, ..., -+90+-d/2]."""
    delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude))
    if (not np.isclose(np.max(latitude), 90 - delta_latitude / 2) or
            not np.isclose(np.min(latitude), -90 + delta_latitude / 2)):
        raise ValueError(
            f'Latitude vector {latitude} does not start/end at '
            '+- (90 - delta_latitude/2) degrees.')
    return np.cos(np.deg2rad(latitude))


def _weight_for_latitude_vector_with_poles(latitude):
    """Weights for uniform latitudes of the form [+- 90, ..., -+90]."""
    delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude))
    if (not np.isclose(np.max(latitude), 90.) or
            not np.isclose(np.min(latitude), -90.)):
        raise ValueError(
            f'Latitude vector {latitude} does not start/end at +- 90 degrees.')
    weights = np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(delta_latitude / 2))
    # The two checks above enough to guarantee that latitudes are sorted, so
    # the extremes are the poles
    weights[[0, -1]] = np.sin(np.deg2rad(delta_latitude / 4)) ** 2
    return weights


def _check_uniform_spacing_and_get_delta(vector):
    diff = np.diff(vector)
    if not np.all(np.isclose(diff[0], diff)):
        raise ValueError(f'Vector {diff} is not uniformly spaced.')
    return diff[0]

def split_grid(h: int, w: int, patch_h: int, patch_w: int) -> list:
    """
    Splits a grid of size (h, w) into patches of size (patch_h, patch_w).

    Args:
        h (int): Height of the grid.
        w (int): Width of the grid.
        patch_h (int): Height of each patch.
        patch_w (int): Width of each patch.

    Returns:
        list: List of tuples representing the patches. Each tuple contains the starting and ending indices of the patch.

    Example:
        split_grid(9, 9, 3, 3) returns [(0, 0, 3, 3), (0, 3, 3, 6), (0, 6, 3, 9), (3, 0, 6, 3), (3, 3, 6, 6), (3, 6, 6, 9), (6, 0, 9, 3), (6, 3, 9, 6), (6, 6, 9, 9)]
    """
    patches = []
    for i in range(0, h, patch_h):
        for j in range(0, w, patch_w):
            starth_h = i
            start_w = j
            end_h = i + patch_h
            end_w = j + patch_w
            patches.append((starth_h, start_w, end_h, end_w))
    return patches


def filter_patches_by_mask(patches, mask):
    filtered_patches = []
    for patch in patches:
        starth_h, start_w, end_h, end_w = patch
        if np.any(mask[starth_h:end_h, start_w:end_w]):
            filtered_patches.append(patch)
    return filtered_patches


def cross_join_patches_with_time(patches, times):
    # cross join patches with times
    patches_times = []
    for patch in patches:
        for time in times:
            patches_times.append((patch, time))
    return patches_times

def load_datasets(ds_path, global_ds_path, input_vars, log_transform_vars, oci_vars, keep_vars=['log_gwis_ba_anom', 'log_gwis_ba', 'gwis_ba', 'gfed_region', 'ndvi', 'pop_dens', 'lsm'], selected_years=None):
    ds = xr.open_zarr(ds_path)
    global_ds = xr.open_zarr(global_ds_path)

    if selected_years is not None:
        ds = ds.sel(time=ds.time.dt.year.isin(selected_years))
        global_ds = global_ds.sel(time=global_ds.time.dt.year.isin(selected_years))
    else:
        selected_years = ds.time.dt.year.values

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
        ds[var] = (ds[var] - ds[var].mean()) / ds[var].std()
        global_ds[var] = (global_ds[var] - global_ds[var].mean()) / global_ds[var].std()

    # calculate additional variables that could be needed.
    ds['log_gwis_ba'] = np.log(ds['gwis_ba'] + 1)
    global_ds['log_gwis_ba'] = np.log(global_ds['gwis_ba'] + 1)

    vegetation_classes = [ 'lccs_class_1', 'lccs_class_2', 'lccs_class_3', 'lccs_class_4',  'lccs_class_6']
    ds['burnable_fraction'] = sum([ds[var].isel(time=0).astype(int) for var in vegetation_classes]) / 100
    ds['gwis_ba_frac'] = ds['gwis_ba'] / (ds['area'] / 10_000)
    ds['gwis_ba_frac'] = ds['gwis_ba_frac'].where(ds['gwis_ba_frac'] < 1e10).where(ds['burnable_fraction']>=0)

    global_ds['burnable_fraction'] = sum([global_ds[var].isel(time=0).astype(int) for var in vegetation_classes]) / 100
    global_ds['gwis_ba_frac'] = global_ds['gwis_ba'] / (global_ds['area'] / 10_000)
    global_ds['gwis_ba_frac'] = global_ds['gwis_ba_frac'].where(global_ds['gwis_ba_frac'] < 1e10).where(global_ds['burnable_fraction']>=0)
    
    # concatenate all vars to keep
    vars_to_keep = keep_vars + input_vars + oci_vars
    # remove duplicates and vars that are not in ds
    vars_to_keep = [x for x in list(set(vars_to_keep)) if x in ds.data_vars]
    # ds['gwis_ba_frac'] = ds['gwis_ba'] / ds['area']

    # load into memory
    ds = ds[vars_to_keep].load()
    global_ds = global_ds[[x for x in vars_to_keep if x in global_ds.data_vars]].load()

    oci_ds = xr.Dataset()
    for var in oci_vars:
        # resample var to 1 month
        oci_ds[var] = ds[var].fillna(0).resample(time='1M').mean(dim='time')
        # normalize oci variables
        oci_ds[var] = (oci_ds[var] - oci_ds[var].mean()) / oci_ds[var].std()

    oci_ds.load()

    ds = _add_positional_vars(ds)
    global_ds = _add_positional_vars(global_ds)


    return ds, global_ds, oci_ds


def calculate_choice_probs(l, temperature, num_choices):
    """
    Calculate the probabilities for each choice based on the given parameters.

    Args:
        l (float): The lambda value.
        temperature (float): The temperature value.
        num_choices (int): The number of choices.

    Returns:
        list: A list of probabilities for each choice.
    """
    probs = [(l/temperature)*math.e**(-(l*x/temperature)) for x in np.arange(1, num_choices+1)]
    norm_probs =  sum([(l/temperature)*math.e**(-((l/temperature)*x)) for x in np.arange(1, num_choices+1)])
    return [x/norm_probs for x in probs]

class LocalGlobalOciDataset(Dataset):
    """
    Dataset class for global-local dataset.

    Args:
        ds_local (xarray.Dataset): Local dataset.
        ds_global (xarray.Dataset): Global dataset.
        ds_oci (xarray.Dataset): OCI dataset.
        input_vars (list): List of input variable names.
        positional_vars (list): List of positional variable names.
        oci_vars (list): List of OCI variable names.
        target (str): Target variable name.
        min_lead_time (int): Minimum lead time.
        max_lead_time (int): Maximum lead time.
        local_lag (int): Local lag.
        oci_lag (int): OCI lag.
        global_lag (int): Global lag.
        patches (list): List of patches.
        task (str, optional): Task type, either 'classification' or 'regression'. Defaults to 'classification'.
        nanfill (float, optional): Value to fill NaNs with. Defaults to -1.0.
    """
    def __init__(self, ds_local, ds_global, ds_oci, input_vars, positional_vars, oci_vars, target, min_lead_time, 
                 max_lead_time, local_lag, oci_lag, global_lag, patches, task='classification', nanfill=-1., 
                 patch_local_shape=None, patch_global_shape=None, posemb_path=None, posemb_global_path=None, posemb_var_name=None, posemb_mask='lsm'):
        self.task = task
        assert self.task in ['classification', 'regression']
        self.positional_vars = positional_vars
        self.patches = patches
        self.target = target
        self.input_vars = input_vars
        self.oci_vars = oci_vars
        self.ds_local = ds_local
        self.ds_oci = ds_oci
        self.ds_global = ds_global
        self.nanfill = nanfill
        self.local_lag = local_lag
        self.oci_lag = oci_lag
        self.global_lag = global_lag
        self.min_lead_time = min_lead_time
        self.max_lead_time = max_lead_time
        self.temperature = 1
        self.num_choices = self.max_lead_time - self.min_lead_time + 1
        self.l = np.log(10) / self.num_choices
        self.probs = calculate_choice_probs(self.l, self.temperature, self.num_choices)
        if len(list(patch_local_shape))> 2:
            self.patch_local_shape = patch_local_shape
        else:
            self.patch_local_shape = [1, 16, 16]
        if len(list(patch_global_shape)) > 2:
            self.patch_global_shape = patch_global_shape
        else:
            self.patch_global_shape = [1, 60, 60]


        self.posemb_path = posemb_path
        if self.posemb_path:
            if posemb_mask=='lsm':
                self.ds_satclip_local = xr.open_zarr(posemb_path)[posemb_var_name].where(ds_local.lsm > 0).load()
                self.global_pos_embedding = xr.open_zarr(posemb_global_path).where(ds_global.lsm > 0).coarsen(dim={'longitude' : self.patch_global_shape[-2], 'latitude': self.patch_global_shape[-1] }).mean(skipna=True).fillna(0)[posemb_var_name].values
            else:
                self.ds_satclip_local = xr.open_zarr(posemb_path)[posemb_var_name].load()
                self.global_pos_embedding = xr.open_zarr(posemb_global_path).coarsen(dim={'longitude' : self.patch_global_shape[-2], 'latitude': self.patch_global_shape[-1] }).mean(skipna=True).fillna(0)[posemb_var_name].values
        self.posemb_var_name = posemb_var_name
        self.ds_local['normalized_weights'] = normalized_latitude_weights(self.ds_local) * xr.ones_like(self.ds_local['area'])

    def __len__(self):
        return len(self.patches)

    def update_temperature(self, increment=1., factor=1.):
        self.temperature += increment
        self.temperature *= factor
        self.probs = calculate_choice_probs(self.l, self.temperature, self.num_choices)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        (start_h, start_w, end_h, end_w), time = patch
        
        if self.min_lead_time == self.max_lead_time:
            lead_time = self.min_lead_time
        else:
            lead_time = np.random.choice(np.arange(self.min_lead_time, self.max_lead_time+1), 1, p=self.probs)
            if idx == 0:
                print()
                print(f'Probabilities: {self.probs}')
                print()

        local_input_ds = self.ds_local.isel(longitude=slice(start_w, end_w), latitude=slice(start_h, end_h),
                                         time=slice(time - self.local_lag - lead_time + 1, time - lead_time + 1))
        local_input = np.stack([local_input_ds[var] for var in self.input_vars], axis=0)
        local_target = self.ds_local.isel(longitude=slice(start_w, end_w), latitude=slice(start_h, end_h),
                                        time=time)[self.target].values
        local_pos = np.stack([local_input_ds[var].values for var in self.positional_vars], axis=0)
        local_input = np.nan_to_num(local_input, nan=self.nanfill)
        local_target = np.squeeze(np.nan_to_num(local_target, nan=0))
        local_mask = np.isnan(local_input_ds.isel(time=-1)['ndvi']).values
        local_burnable_fraction = local_input_ds.isel(time=-1)['burnable_fraction'].values

        oci_ds =  self.ds_oci.sel(time=slice(local_input_ds.time[-1] - np.timedelta64(self.oci_lag * 31, 'D'), local_input_ds.time[-1]))
        oci_input = oci_ds.isel(time=slice(- self.oci_lag, None))[self.oci_vars]
        oci_input = np.stack([oci_input[var] for var in self.oci_vars], axis=0)
        oci_input = np.nan_to_num(oci_input, nan=self.nanfill)

        global_input = self.ds_global.isel(time=slice(time - self.global_lag - lead_time + 1, time - lead_time + 1))
        global_input = np.stack([global_input[var] for var in self.input_vars], axis=0)
        global_input = np.nan_to_num(global_input, nan=self.nanfill)
        global_target = np.squeeze(self.ds_global.isel(time=time)[self.target].values)
        global_pos = np.stack([self.ds_global[var].values for var in self.positional_vars], axis=0)

        if self.task == 'classification':
            local_target = np.where(local_target > 0, 1, 0)
            global_target = np.where(global_target > 0, 1, 0)
        elif self.task == 'regression':
            local_target = np.nan_to_num(local_target, nan=0)
            global_target = np.nan_to_num(global_target, nan=0)

        if self.posemb_path:
            x_local_satclip = self.ds_satclip_local.isel(
                longitude=slice(start_w, end_w), 
                latitude=slice(start_h, end_h)
            ).coarsen(
                dim={'longitude': self.patch_local_shape[-2], 'latitude': self.patch_local_shape[-1]}
            ).mean(skipna=True).fillna(0).values

            x_global_satclip = self.global_pos_embedding
        else:
            x_local_satclip = torch.empty(0)
            x_global_satclip = torch.empty(0)

        return {
            'x_local_satclip': x_local_satclip,
            'x_global_satclip': x_global_satclip,
            'x_local': local_input,
            'x_local_mask': local_mask,
            'x_local_pos': local_pos,
            'x_oci': oci_input,
            'x_global': global_input,
            'x_global_pos': global_pos,
            'y_local': local_target,
            'y_global': global_target,
            'lead_time': lead_time,
            'burnable_fraction': local_burnable_fraction,
            'normalized_weights': local_input_ds['normalized_weights'].values
        }


def get_loaded_datasets(ds_path, global_ds_path, input_vars, log_transform_vars, 
                        oci_vars, keep_vars=['log_gwis_ba',  'gwis_ba', 'gfed_region', 'ndvi', 'pop_dens'], selected_years=None):
    ds, global_ds, oci_ds = load_datasets(ds_path, global_ds_path, input_vars, log_transform_vars, oci_vars, keep_vars=keep_vars, selected_years=selected_years)
    return ds, global_ds, oci_ds

def create_dataset_for_years(ds, global_ds, oci_ds, input_vars, oci_vars, positional_vars, target,
                     min_lead_time, max_lead_time, local_lag, oci_lag, global_lag, patch_size, years, 
                     task='classification', nanfill=-1., patch_local_shape=None, patch_global_shape=None,
                     posemb_path=None, posemb_global_path=None, posemb_var_name=None, posemb_mask='lsm'):
    # load datasets
    # ds, global_ds, oci_ds = load_datasets(ds_path, global_ds_path, input_vars, log_transform_vars, oci_vars, keep_vars=['gwis_ba', 'gfed_region', 'ndvi', 'pop_dens'])

    # split grid into patches
    patch_h, patch_w = patch_size[-2], patch_size[-1]
    patches = split_grid(ds.latitude.size, ds.longitude.size, patch_h, patch_w)

    mask = ds.gfed_region > 0 # mask of areas with inside the gfed region

    # filter patches by mask
    patches = filter_patches_by_mask(patches, mask)

    # select indices of ds.time that are in years
    time_indices = [i for i, time in enumerate(ds.time) if time.dt.year in years]

    # cross join patches with times
    patches_times = cross_join_patches_with_time(patches, time_indices)

    # create dataset
    dataset = LocalGlobalOciDataset(ds, global_ds, oci_ds, input_vars, positional_vars, oci_vars, target, min_lead_time, max_lead_time,
                                     local_lag, oci_lag, global_lag, patches_times, task=task, nanfill=nanfill, 
                                     patch_local_shape=patch_local_shape, patch_global_shape=patch_global_shape,
                                     posemb_path=posemb_path, posemb_global_path=posemb_global_path, posemb_var_name=posemb_var_name, posemb_mask=posemb_mask)

    return dataset

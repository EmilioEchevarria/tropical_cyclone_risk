#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jzlin@mit.edu
"""

import dask
import datetime
import os
import namelist
import numpy as np
import xarray as xr

from dask.distributed import LocalCluster, Client
from util import input, mat
from thermo import thermo

def get_fn_thermo():
    fn_th = '%s/thermo_%s_%d%02d_%d%02d.nc' % (namelist.output_directory, namelist.exp_prefix,
                                               namelist.start_year, namelist.start_month,
                                               namelist.end_year, namelist.end_month)
    return(fn_th)

"""
Path to the per-year thermo cache file
"""
def _get_fn_thermo_yearly(year):
    return '%s/yearly/thermo_%s_%d.nc' % (namelist.output_directory, namelist.exp_prefix, year)


def compute_thermo(dt_start, dt_end):
    ds_sst = input.load_sst(dt_start, dt_end).load()
    ds_psl = input.load_mslp(dt_start, dt_end).load()
    ds_ta = input.load_temp(dt_start, dt_end).load()
    ds_hus = input.load_sp_hum(dt_start, dt_end).load()
    lon_ky = input.get_lon_key()
    lat_ky = input.get_lat_key()
    sst_ky = input.get_sst_key()

    nTime = len(ds_sst['time'])
    vmax = np.zeros(ds_psl[input.get_mslp_key()].shape)
    chi = np.zeros(ds_psl[input.get_mslp_key()].shape)
    rh_mid = np.zeros(ds_psl[input.get_mslp_key()].shape)
    for i in range(nTime):
        # Convert all variables to the atmospheric grid.
        sst_interp = mat.interp_2d_grid(ds_sst[lon_ky], ds_sst[lat_ky],
                                        np.nan_to_num(ds_sst[sst_ky][i, :, :].data),
                                        ds_ta[lon_ky], ds_ta[lat_ky])
        if 'C' in ds_sst[sst_ky].units:
            sst_interp = sst_interp + 273.15

        psl = ds_psl[input.get_mslp_key()][i, :, :]
        ta = ds_ta[input.get_temp_key()][i, :, :, :]
        hus = ds_hus[input.get_sp_hum_key()][i, :, :, :]
        lvl = ds_ta[input.get_lvl_key()]
        lvl_d = np.copy(ds_ta[input.get_lvl_key()].data)

        # Ensure lowest model level is first.
        # Here we assume the model levels are in pressure.
        if (lvl[0] - lvl[1]) < 0:
            ta = ta.reindex({input.get_lvl_key(): lvl[::-1]})
            hus = hus.reindex({input.get_lvl_key(): lvl[::-1]})
            lvl_d = lvl_d[::-1]
    
        p_midlevel = namelist.p_midlevel                    # Pa
        if lvl.units in ['millibars', 'hPa']:
            lvl_d *= 100                                    # needs to be in Pa
            p_midlevel = namelist.p_midlevel / 100          # hPa
            lvl_mid = lvl.sel({input.get_lvl_key(): p_midlevel}, method = 'nearest')

        # TODO: Check units of psl, ta, and hus
        vmax_args = (sst_interp, psl.data, lvl_d, ta.data, hus.data)
        vmax[i, :, :] = thermo.CAPE_PI_vectorized(*vmax_args)
        ta_midlevel = ta.sel({input.get_lvl_key(): p_midlevel}, method = 'nearest').data
        hus_midlevel = hus.sel({input.get_lvl_key(): p_midlevel}, method = 'nearest').data

        p_midlevel_Pa = float(lvl_mid) * 100 if lvl_mid.units in ['millibars', 'hPa'] else float(lvl_mid)
        chi_args = (sst_interp, psl.data, ta_midlevel,
                    p_midlevel_Pa, hus_midlevel)
        chi[i, :, :] = np.minimum(np.maximum(thermo.sat_deficit(*chi_args), 0), 10)
        rh_mid[i, :, :] = thermo.conv_q_to_rh(ta_midlevel, hus_midlevel, p_midlevel_Pa)

    return (vmax, chi, rh_mid)

def _compute_and_save_year(year, ds_ref):
    """Compute thermo for a single year and write its yearly cache file"""
    import xarray as _xr
    dt_start = datetime.datetime(year, 1, 1)
    dt_end = datetime.datetime(year, 12, 31, 23, 59)

    # Grab the month-centered timestamps that fall in this year
    times_year = [x for x in input.convert_to_datetime(ds_ref, ds_ref['time'].values) if x >= dt_start and x <= dt_end]
    if len(times_year) == 0:
        return None

    vmax, chi, rh_mid = compute_thermo(times_year[0], times_year[-1])

    ds_times_year = input.convert_from_datetime(ds_ref, np.array([datetime.datetime(x.year, x.month, 15) for x in times_year]))

    lat_full = ds_ref[input.get_lat_key()].data
    lat_mask = (lat_full >= -65) & (lat_full <= 65)
    lat_filt = lat_full[lat_mask]
    vmax = vmax[:, lat_mask, :]
    chi = chi[:, lat_mask, :]
    rh_mid = rh_mid[:, lat_mask, :]

    ds_thermo = _xr.Dataset(
        data_vars=dict(vmax=(['time', 'lat', 'lon'], vmax),
                       chi=(['time', 'lat', 'lon'], chi),
                       rh_mid=(['time', 'lat', 'lon'], rh_mid)),
        coords=dict(lon=("lon", ds_ref[input.get_lon_key()].data),
                    lat=("lat", lat_filt),
                    time=("time", ds_times_year.astype('datetime64[ns]'))))
    fn_yr = _get_fn_thermo_yearly(year)
    os.makedirs(os.path.dirname(fn_yr), exist_ok=True)

    try:
        ds_thermo.to_netcdf(fn_yr, engine='h5netcdf')
        print('Saved yearly cache %s' % fn_yr, flush=True)
    except Exception:
        ds_thermo.to_netcdf(fn_yr, engine='netcdf4')
        print('Saved yearly cache %s' % fn_yr, flush=True)
    
    return fn_yr

def gen_thermo():
    # TODO: Assert all of the datasets have the same length in time.
    if os.path.exists(get_fn_thermo()):
        return

    os.makedirs('%s/yearly' % namelist.output_directory, exist_ok=True)
    dt_start, dt_end = input.get_bounding_times()
    ds = input.load_mslp()

    # Years for which a yearly cache must be produced
    years_needed = list(range(namelist.start_year, namelist.end_year + 1))
    years_to_compute = [y for y in years_needed if not os.path.exists(_get_fn_thermo_yearly(y))]
    years_cached = [y for y in years_needed if y not in years_to_compute]
    if years_cached:
        print('[gen_thermo] yearly cache hit for years: %s' % years_cached, flush=True)

    if years_to_compute:
        if namelist.use_dask:
            dask.config.set({'distributed.worker.profile.enabled': False})
            n_workers = min(namelist.n_procs, len(years_to_compute))
            cl_args = {'n_workers': n_workers,
                       'processes': True,
                       'threads_per_worker': 1}
            with LocalCluster(**cl_args) as cluster, Client(cluster) as client:
                lazy_results = [dask.delayed(_compute_and_save_year)(y, ds)
                                for y in years_to_compute]
                dask.compute(*lazy_results, scheduler='processes',
                             num_workers=n_workers)
        else:
            for y in years_to_compute:
                _compute_and_save_year(y, ds)

    # Assemble the combined file from yearly caches for the requested range
    yearly_fns = [_get_fn_thermo_yearly(y) for y in years_needed if os.path.exists(_get_fn_thermo_yearly(y))]
    if not yearly_fns:
        raise RuntimeError('No yearly thermo caches available to assemble %s' % get_fn_thermo())

    ds_combined = xr.open_mfdataset(yearly_fns, combine='by_coords').load()

    try:
        ds_combined.to_netcdf(get_fn_thermo(), engine='h5netcdf')
        ds_combined.close()
    except Exception:
        ds_combined.to_netcdf(get_fn_thermo(), engine='netcdf4')
        ds_combined.close()
    print('Saved %s' % get_fn_thermo())

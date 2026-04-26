"""
@author: craig.arthur@ga.gov.au
"""

import dask
import datetime
import os

import dask.delayed
import namelist
import numpy as np
import pandas as pd
import xarray as xr
import metpy.calc as mpcalc
import metpy.constants as mpconst
from util import input, mat
from thermo import thermo, calc_thermo
from genesis import genesis
from track import env_wind


def get_fn_tcgp():
    """
    Retrieve the filename of the TC genesis parameter data file

    :return: Formatted file name
    :rtype: str
    """
    fn_tcgp = "%s/tcgp_%s_%d%02d_%d%02d.nc" % (
        namelist.output_directory,
        namelist.exp_prefix,
        namelist.start_year,
        namelist.start_month,
        namelist.end_year,
        namelist.end_month,
    )
    return fn_tcgp

def _get_fn_tcgp_yearly(year):
    """Path to the per-year tcgp cache file."""
    return '%s/yearly/tcgp_%s_%d.nc' % (namelist.output_directory, namelist.exp_prefix, year)

def genesis_point(tcgp, size=1):
    """
    Return a genesis point, based on weighted random sampling

    :param tcgp: `xr.DataArray` of TC genesis parameter
    :param size: number of samples to return, defaults to 1

    :return: Longitude and latitude of genesis point
    :rtype: list
    """

    interplon = np.arange(tcgp.longitude.min(), tcgp.longitude.max()+0.01, 0.01)
    interplat = np.arange(tcgp.latitude.min(), tcgp.latitude.max()+0.01, 0.01)

    # Step 2: Interpolate the weights onto the finer grid
    # Using xarray's interp method with method='linear' to interpolate NaNs and refine the grid
    tcgp_interp = tcgp.interp(longitude=interplon, latitude=interplat, method='linear')

    tcp = tcgp_interp.values.flatten()
    tcp = np.nan_to_num(tcp, nan=0)
    weights = tcp / np.nansum(tcp)
    weights = np.where(weights < 0., 0., weights)
    idx = np.random.choice(len(tcp), size=size, p=weights)
    idx2d = [np.unravel_index(index, tcgp_interp.shape) for index in idx]
    genlatlon = np.array([[tcgp_interp.latitude[lat_idx].item(), tcgp_interp.longitude[lon_idx].item()]
                         for lat_idx, lon_idx in idx2d]).flatten()

    if size > 1:
        genlat = genlatlon[0, :]
        genlon = genlatlon[1, :]
    else:
        genlat = genlatlon[0]
        genlon = genlatlon[1]
    return genlon, genlat


def compute_genesis(dt_start, dt_end):
    """
    Compute genesis parameter, based on potential intensity, mid level
    humidity, wind shear and normalised vorticity

    :param dt_start: Date/time of the start of the simulation
    :type dt_start: :class:`datetime.datetime`

    :param dt_end: Date/time of the end of the simulation
    :type dt_end: :class:`datetime.datetime`

    :return: array of TC genesis parameter
    :rtype: numpy array
    """
    
    fn_th = calc_thermo.get_fn_thermo()
    fn_wnd_stat = env_wind.get_env_wnd_fn()

    thermo_ds = xr.open_dataset(fn_th).sel(time=slice(dt_start, dt_end)).sortby("lat", ascending=True).sel(lat=slice(-65,65))
    wnd_ds = xr.open_dataset(fn_wnd_stat).sel(time=slice(dt_start, dt_end)).sortby("lat", ascending=True).sel(lat=slice(-65,65))
    
    xi = genesis._xi(wnd_ds)
    shear = genesis._shear(wnd_ds)
    vpot = thermo_ds['vmax'].sel(time=slice(dt_start, dt_end))
    ds_ta = input.load_temp(dt_start, dt_end).sortby("latitude", ascending=True).sel(latitude=slice(-65,65)).load()
    ds_hus = input.load_sp_hum(dt_start, dt_end).sortby("latitude", ascending=True).sel(latitude=slice(-65,65)).load()

    ds_ta = ds_ta.sel(time=vpot.time.values, method='nearest')
    ds_hus = ds_hus.sel(time=vpot.time.values, method='nearest')

    ta = ds_ta[input.get_temp_key()]
    hus = ds_hus[input.get_sp_hum_key()]
    lvl = ds_ta[input.get_lvl_key()]
    lvl_d = np.copy(ds_ta[input.get_lvl_key()].data)

    # For genesis potential, we use 700 hPa relative humidity
    p_midlevel_rh = namelist.genesis_rh_level           # hPa
    if lvl.units in ['millibars', 'hPa']:
        lvl_d *= 100                                    # needs to be in Pa
        lvl_mid = lvl.sel({input.get_lvl_key(): p_midlevel_rh},
                          method = 'nearest')
    ta_midlevel = ta.sel({input.get_lvl_key(): p_midlevel_rh},
                         method = 'nearest').data
    hus_midlevel = hus.sel({input.get_lvl_key(): p_midlevel_rh},
                           method = 'nearest').data
    
    p_midlevel_Pa = float(lvl_mid) * 100 if lvl_mid.units in ['millibars', 'hPa'] else float(lvl_mid)

    rh_mid_data = thermo.conv_q_to_rh(ta_midlevel, hus_midlevel, p_midlevel_Pa)
    rh_mid = xr.DataArray(
        data=rh_mid_data,
        dims=vpot.dims,
        coords={d: vpot.coords[d] for d in vpot.dims if d in vpot.coords},
        name='rh_mid',
        )

    tcgp = genesis._tcgp(vpot, xi, rh_mid, shear)
    
    return tcgp

def _compute_and_save_tcgp_year(year, ds_ref):
    """Compute tcgp for a single year and write its yearly cache file."""
    import xarray as _xr
    # Grab the month-centered timestamps that fall in this year
    times_year = [
        x for x in input.convert_to_datetime(ds_ref, ds_ref["time"].values)
        if x >= datetime.datetime(year, 1, 1) and x <= datetime.datetime(year, 12, 31, 23, 59)
    ]
    if len(times_year) == 0:
        return None

    # Extend end so day=15 monthly thermo/wnd entries fall inside the slice
    dt_start = times_year[0]
    dt_end_padded = (np.datetime64(times_year[-1]) + np.timedelta64(20, 'D')).astype('datetime64[us]').astype(datetime.datetime)

    tcgp_yr = compute_genesis(dt_start, dt_end_padded)

    # Timestamps at day=15 for storage
    ds_times_year = input.convert_from_datetime(
        ds_ref,
        np.array([datetime.datetime(x.year, x.month, 15) for x in times_year]))

    lat_full = ds_ref[input.get_lat_key()].data
    lat_sorted = np.sort(lat_full)
    lat_filt = lat_sorted[(lat_sorted >= -65) & (lat_sorted <= 65)]

    ds_genesis = _xr.Dataset(
        data_vars=dict(tcgp=(["time", "lat", "lon"], tcgp_yr.data)),
        coords=dict(lon=("lon", ds_ref[input.get_lon_key()].data),
                    lat=("lat", lat_filt),
                    time=("time", ds_times_year)))
    fn_yr = _get_fn_tcgp_yearly(year)
    os.makedirs(os.path.dirname(fn_yr), exist_ok=True)
    try:
        ds_genesis.to_netcdf(fn_yr, engine='h5netcdf')
    except Exception:
        ds_genesis.to_netcdf(fn_yr, engine='netcdf4')
    print('Saved yearly cache %s' % fn_yr, flush=True)
    return fn_yr

def gen_genesis():
    if os.path.exists(get_fn_tcgp()):
        return

    os.makedirs('%s/yearly' % namelist.output_directory, exist_ok=True)
    ds = input.load_mslp().sortby("latitude", ascending=True).sel(latitude=slice(-65, 65))

    years_needed = list(range(namelist.start_year, namelist.end_year + 1))
    years_to_compute = [y for y in years_needed
                        if not os.path.exists(_get_fn_tcgp_yearly(y))]
    years_cached = [y for y in years_needed if y not in years_to_compute]
    if years_cached:
        print('[gen_genesis] yearly cache hit for years: %s' % years_cached, flush=True)

    for i, y in enumerate(years_to_compute):
        print(f"[gen_genesis] Processing year {y} ({i+1}/{len(years_to_compute)})...", flush=True)
        _compute_and_save_tcgp_year(y, ds)

    # Assemble combined from yearly caches for the requested range.
    yearly_fns = [_get_fn_tcgp_yearly(y) for y in years_needed if os.path.exists(_get_fn_tcgp_yearly(y))]
    if not yearly_fns:
        raise RuntimeError('No yearly tcgp caches available to assemble %s' % get_fn_tcgp())

    ds_combined = xr.open_mfdataset(yearly_fns, combine='by_coords').load()

    try:
        ds_combined.to_netcdf(get_fn_tcgp(), engine='h5netcdf')
    except Exception:
        ds_combined.to_netcdf(get_fn_tcgp(), engine='netcdf4')
    print("Saved %s" % get_fn_tcgp())

"""
Environment-dependent density of genesis seeds.

With namelist.seed_density = True, seeds are accepted in proportion to the local frequency of 'seed days'
instead of being spread uniformly and discarded towards the equator with basin-tuned params. A seed day
is a day on which the flow can host a TC, by the Okubo-Weiss-Zeta (OWZ) criteria of Tory et al. (2013): 
OWZ850 >= T, OWZ700 >= 0.8 T, cyclonic vorticity at 850 and 700 hPa, and
|V250 - V850| < shear_max, with OWZ = sgn(f) * max(0, (zeta^2 - E^2 - F^2) / zeta^2) * (zeta + f)
(zeta the relative vorticity, E and F the deformation). Only winds enter: potential intensity, humidity,
ventilation and the ocean still decide which seeds become storms.

The storms found by util/tc_tracker.py masked within namelist.seed_tc_radius_km
"""

import datetime
import os
import warnings

import dask
import numpy as np
import xarray as xr
from dask.distributed import LocalCluster, Client
from scipy.ndimage import gaussian_filter

import namelist
from util import input, mat

""" Returns whether environment-dependent seeding is switched on in the namelist """
def enabled():
    return bool(getattr(namelist, 'seed_density', False))

def get_fn_seed():
    fn_out = '%s/seed_%s_%d%02d_%d%02d.nc' % (namelist.output_directory, namelist.exp_prefix,
                                              namelist.start_year, namelist.start_month,
                                              namelist.end_year, namelist.end_month)
    return(fn_out)

"""
Cyclonic relative vorticity and OWZ (1/s) of winds (..., lat, lon); latitude in degrees (any order),
periodic longitude.
"""
_R_Earth = 6.371e6
def owz(u, v, lat, lon):
    phi = np.radians(lat)[:, None]
    dx = _R_Earth * np.cos(phi) * np.radians(abs(lon[1] - lon[0]))
    ux = (np.roll(u, -1, -1) - np.roll(u, 1, -1)) / (2 * np.maximum(dx, 1.0))
    vx = (np.roll(v, -1, -1) - np.roll(v, 1, -1)) / (2 * np.maximum(dx, 1.0))
    y = _R_Earth * np.radians(lat)
    uy = np.gradient(u, y, axis=-2)
    vy = np.gradient(v, y, axis=-2)
    zeta = vx - uy + u * np.tan(phi) / _R_Earth
    z2 = zeta ** 2
    own = np.where(z2 > 1e-14, np.clip((z2 - (ux - vy) ** 2 - (vx + uy) ** 2) / np.maximum(z2, 1e-14), 0.0, 1.0), 0.0)
    sg = np.sign(np.sin(phi))
    return zeta * sg, sg * own * (zeta + 2 * 7.292e-5 * np.sin(phi)) # omega = 7.292e-5

"""
Days (datetime64[D]), lons and lats of the TCs found by util/tc_tracker.py
"""
def storm_positions():
    from util import tc_tracker
    z = np.load(tc_tracker.get_fn_positions())
    return (z['day'], z['lon'], z['lat'])

""" Boolean (day, lat, lon) mask, True within radius_km of any of the positions on that day """
def storm_mask(days, lat, lon, pos, radius_km):
    from scipy.spatial import cKDTree
    tc_day, tc_lon, tc_lat = pos
    LAT, LON = np.meshgrid(np.radians(lat), np.radians(lon % 360.0), indexing = 'ij')
    xyz = np.c_[(np.cos(LAT) * np.cos(LON)).ravel(), (np.cos(LAT) * np.sin(LON)).ravel(), np.sin(LAT).ravel()]
    chord = 2 * np.sin(radius_km * 1000.0 / _R_Earth / 2)
    near = np.zeros((len(days), len(lat), len(lon)), bool)
    for di, d in enumerate(days):
        j = np.nonzero(tc_day == d)[0]
        if len(j) == 0:
            continue
        p1, l1 = np.radians(tc_lat[j]), np.radians(tc_lon[j])
        tree = cKDTree(np.c_[np.cos(p1) * np.cos(l1), np.cos(p1) * np.sin(l1), np.sin(p1)])
        dist, _ = tree.query(xyz, k = 1, distance_upper_bound = chord)
        near[di] = np.isfinite(dist).reshape(len(lat), len(lon))
    return near

"""
For the months of one pair of daily wind files: the fraction of usable days (away from observed storms) that
are seed days, and the fraction of days that are usable.
"""
def seed_stat_wrapper(args):
    fn_u, fn_v = args
    dt_start, dt_end = input.get_bounding_times()
    ds_ua = input._load_var_daily(fn_u)
    ds_va = input._load_var_daily(fn_v)
    ua = ds_ua[input.get_u_key()]
    va = ds_va[input.get_v_key()]
    lat = ua[input.get_lat_key()].data
    lon = ua[input.get_lon_key()].data
    lvl = ua[input.get_lvl_key()]
    p_fac = 1 if lvl.units in ['millibars', 'hPa'] else 100

    dts = input.convert_to_datetime(ds_ua, ds_ua['time'].values)
    t_months = []
    for dt in dts:
        if (dt >= datetime.datetime(dt_start.year, dt_start.month, 1)) and (dt <= dt_end):
            m15 = datetime.datetime(dt.year, dt.month, 15)
            if m15 not in t_months:
                t_months.append(m15)
    if len(t_months) == 0:
        return

    pos = storm_positions()
    thr = namelist.seed_owz_threshold
    out = np.zeros((len(t_months), len(lat), len(lon)), np.float32)
    usable_frac = np.ones((len(t_months), len(lat), len(lon)), np.float32)
    for i, tm in enumerate(t_months):
        k = np.nonzero([(dt.year == tm.year) and (dt.month == tm.month) for dt in dts])[0]
        days = np.array([np.datetime64(dts[j].date()) for j in k])
        uniq, inv = np.unique(days, return_inverse = True)
        # Daily means (in case the files are sub-daily).
        def daily(da, p):
            x = da.sel({input.get_lvl_key(): p * p_fac}, method = 'nearest').isel(time = k).data.astype(np.float64)
            return np.stack([x[inv == n].mean(axis = 0) for n in range(len(uniq))])
        u250, v250 = daily(ua, 250), daily(va, 250)
        u700, v700 = daily(ua, 700), daily(va, 700)
        u850, v850 = daily(ua, 850), daily(va, 850)
        zc850, owz850 = owz(u850, v850, lat, lon)
        zc700, owz700 = owz(u700, v700, lat, lon)
        seed_day = ((owz850 >= thr) & (owz700 >= 0.8 * thr) & (zc850 > 0) & (zc700 > 0) &
                    (np.hypot(u250 - u850, v250 - v850) < namelist.seed_shear_max_ms))
        usable = ~storm_mask(uniq, lat, lon, pos, namelist.seed_tc_radius_km)
        out[i] = (seed_day & usable).sum(axis = 0) / np.maximum(usable.sum(axis = 0), 1)
        usable_frac[i] = usable.mean(axis = 0)

    coords = dict(lon = ('lon', lon), lat = ('lat', lat),
                  time = ('time', input.convert_from_datetime(ds_ua, t_months).astype('datetime64[ns]')))
    ds = xr.Dataset({'seed_frac': (['time', 'lat', 'lon'], out), 'usable': (['time', 'lat', 'lon'], usable_frac)},
                    coords = coords)
    fn = '%s/seed_%s_p%d%02d_%d%02d.nc' % (namelist.output_directory, namelist.exp_prefix,
                                           t_months[0].year, t_months[0].month,
                                           t_months[-1].year, t_months[-1].month)
    ds.to_netcdf(fn)
    return fn

""" Computes the monthly seed-day frequency for the run's period and saves it to get_fn_seed(). """
def gen_seed_density():
    fn_out = get_fn_seed()
    if os.path.exists(fn_out):
        return
    fns_ua = input._glob_prefix(input.get_u_key())
    fns_va = input._glob_prefix(input.get_v_key())
    cl_args = {'n_workers': namelist.n_procs,
               'processes': True,
               'threads_per_worker': 1}
    lazy_results = []
    with LocalCluster(**cl_args) as cluster, Client(cluster) as client:
        for i in range(min(len(fns_ua), len(fns_va))):
            lazy_results.append(dask.delayed(seed_stat_wrapper)((fns_ua[i], fns_va[i])))
        out = dask.compute(*lazy_results)
        out_fns = [x for x in out if x is not None]
    ds = input._open_fns(out_fns).load()
    # Days near observed storms count at the long-term seed-day rate of that place and calendar month, so
    # neither the storms nor the gaps they leave set the seed field of a particular year.
    hits = ds['seed_frac'] * ds['usable']
    month = ds['time'].dt.month
    clim = hits.groupby(month).sum('time') / np.maximum(ds['usable'].groupby(month).sum('time'), 1e-9)
    ds['seed_days'] = hits + clim.sel(month = month).drop_vars('month') * (1 - ds['usable'])
    ds.attrs['note'] = 'Monthly fraction of days on which the flow can host a deep, rotation-dominated cyclonic vortex (util/seeds.py)'
    ds.to_netcdf(fn_out, mode = 'w')
    ds.close()
    print('Saved %s' % fn_out)
    for fn in out_fns:
        os.remove(fn)

"""
Interpolators (one per month, by month index 0-11) of the smoothed seed-day frequency for the given year,
on the input grid with latitude increasing.
"""
def monthly_interpolators(year):
    ds = xr.open_dataset(get_fn_seed())
    times = input.convert_to_datetime(ds, ds['time'].values)
    lat = ds['lat'].data
    lon = ds['lon'].data
    x = ds['seed_days'].data
    ds.close()
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        x = x[:, ::-1, :]
    sig = getattr(namelist, 'seed_smooth_deg', 1.5) / abs(lat[1] - lat[0])
    fx = [None] * 12
    for i, t in enumerate(times):
        if t.year == year:
            fx[t.month - 1] = mat.interp2_fx(lon, lat, gaussian_filter(np.nan_to_num(x[i]), sig, mode = ('nearest', 'wrap')))
    return fx

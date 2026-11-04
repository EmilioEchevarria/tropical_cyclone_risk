"""
TC tracker that uses only daily winds at 850, 700 and 250 hPa, so the same code runs on a reanalysis
and on a climate model's daily output. Its positions tell track/env_filter.py and util/seeds.py which TCs to
leave out of the wind statistics and the seed field
"""
import datetime
import os

import dask
import numpy as np
import xarray as xr
from dask.distributed import LocalCluster, Client
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d, maximum_filter

import namelist
from util import input, seeds

""" Returns whether the wind filter or the seed field take their TCs from this tracker """
def enabled():
    return getattr(namelist, 'env_wind_tc_radius_km', 0) > 0 or seeds.enabled()

def get_fn_positions():
    fn_out = '%s/tc_positions_%s_%d%02d_%d%02d.npz' % (namelist.output_directory, namelist.exp_prefix,
                                                       namelist.start_year, namelist.start_month,
                                                       namelist.end_year, namelist.end_month)
    return(fn_out)

""" Gaussian smoothing of (day, lat, lon) on the sphere: sig_deg of latitude, sig_deg / cos(latitude) of longitude """
def _smooth(x, lat, sig_deg):
    d = abs(lat[1] - lat[0])
    y = gaussian_filter1d(x, sig_deg / d, axis = 1, mode = 'nearest')
    for i, c in enumerate(np.cos(np.radians(lat))):
        y[:, i] = gaussian_filter1d(y[:, i], min(sig_deg / d / max(c, 0.1), 60.0), axis = -1, mode = 'wrap')
    return y

""" Largest value within size grid cells around each point, per day, periodic in longitude. """
def _box_max(x, size):
    return maximum_filter(x, size = (1, size, size), mode = ('nearest', 'nearest', 'wrap'))

"""
Candidate vortices of one pair of daily wind files. Returns the days in the run's period and a table with
one row per candidate (see COLUMNS), iday counting those days
"""
COLUMNS = ['iday', 'lon', 'lat', 'zs850', 'zs700', 'zs250', 'owz850', 'owz700', 'wind850', 'shear', 'land']
def scan_file(args):
    fn_u, fn_v = args
    dt_start, dt_end = input.get_bounding_times()
    ds_ua = input._load_var_daily(fn_u)
    ds_va = input._load_var_daily(fn_v)
    ua = ds_ua[input.get_u_key()]
    va = ds_va[input.get_v_key()]
    lat_key, lon_key, lvl_key = input.get_lat_key(), input.get_lon_key(), input.get_lvl_key()
    p_fac = 1 if ua[lvl_key].units in ['millibars', 'hPa'] else 100

    dts = input.convert_to_datetime(ds_ua, ds_ua['time'].values)
    k_all = [i for i, dt in enumerate(dts) if (dt >= datetime.datetime(dt_start.year, dt_start.month, 1)) and (dt <= dt_end)]
    days = np.array([np.datetime64(dts[i].date()) for i in k_all], dtype = 'datetime64[D]')
    uniq, inv = np.unique(days, return_inverse = True)
    k_all = np.array(k_all)
    if len(uniq) == 0:
        return uniq, np.zeros((0, len(COLUMNS)), np.float32)

    lat_in, lon_in = ua[lat_key].data, ua[lon_key].data
    lat, lon = np.sort(lat_in), np.sort(lon_in % 360.0)
    grid = namelist.tc_tracker_grid_deg
    on_grid = (abs(abs(lat_in[1] - lat_in[0]) - grid) < 1e-3) and (abs(abs(lon_in[1] - lon_in[0]) - grid) < 1e-3) and (lon_in.min() >= 0)
    if not on_grid:
        lat = np.arange(np.ceil(lat_in.min()), lat_in.max() + 1e-6, grid)
        lon = np.arange(0.0, 360.0, grid)
    rows = np.abs(lat) <= 55
    lat_r = lat[rows]

    def daily(da, p, n):
        # Daily means on the tracker's grid, latitude increasing.
        k = np.nonzero(np.isin(inv, n))[0]
        x = da.sel({lvl_key: p * p_fac}, method = 'nearest').isel(time = k_all[k])
        coords = {lat_key: lat_r, lon_key: lon}
        x = x.sel(coords) if on_grid else x.interp(coords, kwargs = dict(fill_value = None))
        x = x.transpose('time', lat_key, lon_key).data.astype(np.float64)
        return np.stack([x[inv[k] == m].mean(axis = 0) for m in n])

    ds = xr.open_dataset('%s/intensity/data/land.nc' % namelist.src_directory)
    o = np.argsort(ds['lat'].values)
    f_land = RegularGridInterpolator((ds['lat'].values[o], ds['lon'].values), ds['land'].values[o],
                                     bounds_error = False, fill_value = 0, method = 'nearest')
    ds.close()

    tabs = []
    for n0 in range(0, len(uniq), 31):
        n = np.arange(n0, min(n0 + 31, len(uniq)))
        u250, v250 = daily(ua, 250, n), daily(va, 250, n)
        u700, v700 = daily(ua, 700, n), daily(va, 700, n)
        u850, v850 = daily(ua, 850, n), daily(va, 850, n)
        z850, owz850 = seeds.owz(u850, v850, lat_r, lon)
        z700, owz700 = seeds.owz(u700, v700, lat_r, lon)
        z250, _ = seeds.owz(u250, v250, lat_r, lon)
        zs850, zs700, zs250 = _smooth(z850, lat_r, 1.5), _smooth(z700, lat_r, 1.5), _smooth(z250, lat_r, 1.5)
        shear = np.hypot(_smooth(u250 - u850, lat_r, 5.0), _smooth(v250 - v850, lat_r, 5.0))
        d, i, j = np.nonzero((zs850 == _box_max(zs850, 7)) & (zs850 >= 1e-5) & (np.abs(lat_r)[None, :, None] <= 50))
        if len(d) == 0:
            continue
        o850, o700 = _box_max(owz850, 5), _box_max(owz700, 5)
        w850, z700m = _box_max(np.hypot(u850, v850), 7), _box_max(zs700, 5)
        tabs.append(np.c_[n[d], lon[j], lat_r[i], zs850[d, i, j], z700m[d, i, j], zs250[d, i, j], o850[d, i, j],
                          o700[d, i, j], w850[d, i, j], shear[d, i, j], f_land(np.c_[lat_r[i], lon[j]]) > 0.5])
    ds_ua.close()
    ds_va.close()
    return uniq, np.concatenate(tabs).astype(np.float32)

""" Point(s) on the unit sphere and the straight-line distance between points this far apart """
def _xyz(lon, lat):
    p, l = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(p) * np.cos(l), np.cos(p) * np.sin(l), np.sin(p)]

def _chord(km):
    return 2 * np.sin(km / 6371.0 / 2)

""" Greedy day-to-day links, strongest vortex first, nearest candidate within reach. Returns lists of row numbers """
def _link(c, p):
    days = np.unique(c['iday'])
    by_day = {d: np.nonzero(c['iday'] == d)[0] for d in days}
    open_tr, done = [], []
    for d in range(int(days.min()), int(days.max()) + 1):
        idx = by_day.get(d, np.array([], int))
        if len(idx):
            idx = idx[np.argsort(-c['zs850'][idx])]
        pos = _xyz(c['lon'][idx], c['lat'][idx]) if len(idx) else None
        taken = set()
        new_open = []
        for tr in sorted(open_tr, key = lambda x: -c['zs850'][x[-1]]):
            last = tr[-1]
            gap = d - c['iday'][last]
            got = False
            if len(idx):
                reach = (p['step_km'] if abs(c['lat'][last]) < 30 else p['step_km_high']) * gap
                dist = np.linalg.norm(pos - _xyz(c['lon'][[last]], c['lat'][[last]]), axis = 1)
                for m in np.argsort(dist):
                    if dist[m] > _chord(reach):
                        break
                    if m not in taken:
                        tr.append(idx[m])
                        taken.add(m)
                        got = True
                        break
            if got or (gap < 1 + p['gap_days']):
                new_open.append(tr)
            else:
                done.append(tr)
        for m in range(len(idx)):
            if m not in taken:
                new_open.append([idx[m]])
        open_tr = new_open
    return [np.array(tr) for tr in done + open_tr]

""" The storm stages in a candidate table: a list of (iday, lon, lat) arrays """
def build_tracks(tab):
    p = namelist.tc_tracker_params
    c = {name: tab[:, i].astype(np.float64) for i, name in enumerate(COLUMNS)}
    c['iday'] = c['iday'].astype(int)
    keep = ((c['zs850'] >= p['z_track']) & (c['owz850'] >= p['owz']) & (c['owz700'] >= 0.8 * p['owz']) &
            (c['zs700'] >= p['z700_frac'] * c['zs850']) & (c['zs850'] >= c['zs250']))
    c = {name: x[keep] for name, x in c.items()}
    tracks = []
    for i in _link(c, p):
        strong = (c['zs850'][i] >= p['z_strong']) & (c['wind850'][i] >= p['w_strong']) & (c['land'][i] < 0.5)
        if (len(i) < p['min_days']) or (strong.sum() < p['min_strong']):
            continue
        j = np.nonzero(strong)[0][0]
        if (abs(c['lat'][i][j]) > p['lat0']) or (c['shear'][i][j] > p['shear0']):
            continue
        s = i[j:min(len(i), np.nonzero(strong)[0][-1] + 1 + p['tail_days'])]
        tracks.append((c['iday'][s], c['lon'][s], c['lat'][s]))
    return tracks

"""
Each tracked day's position, returns (iday, lon, lat)
"""
def mask_positions(tracks):
    D, X, A = [], [], []
    for d, x, a in tracks:
        for i in range(len(d)):
            D.append(d[i]); X.append(x[i]); A.append(a[i])
            for j in (i - 1, i + 1):
                if 0 <= j < len(d):
                    gap = abs(d[j] - d[i])
                    dx = (x[j] - x[i] + 180.0) % 360.0 - 180.0
                    for fr in (0.25, 0.5):
                        D.append(d[i]); X.append((x[i] + fr * dx / gap) % 360.0); A.append(a[i] + fr * (a[j] - a[i]) / gap)
    return np.array(D, int), np.array(X, float), np.array(A, float)

""" Finds the storms in the daily winds of the run's period and saves their positions to get_fn_positions(). """
def gen_positions():
    fn_out = get_fn_positions()
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
            lazy_results.append(dask.delayed(scan_file)((fns_ua[i], fns_va[i])))
        out = dask.compute(*lazy_results)
    out = sorted([x for x in out if len(x[0]) > 0], key = lambda x: x[0][0])
    days, tabs, n = [], [], 0
    for d, t in out:
        t = t.copy()
        t[:, 0] += n
        days.append(d)
        tabs.append(t)
        n += len(d)
    days = np.concatenate(days)
    tracks = build_tracks(np.concatenate(tabs))
    d, x, a = mask_positions(tracks)
    track_id = np.concatenate([np.full(len(t[0]), i) for i, t in enumerate(tracks)])
    np.savez_compressed(fn_out, day = days[d], lon = x, lat = a, track_id = track_id,
                        track_day = days[np.concatenate([t[0] for t in tracks])],
                        track_lon = np.concatenate([t[1] for t in tracks]),
                        track_lat = np.concatenate([t[2] for t in tracks]))
    print('Saved %s: %d tracks in %d days' % (fn_out, len(tracks), len(days)))

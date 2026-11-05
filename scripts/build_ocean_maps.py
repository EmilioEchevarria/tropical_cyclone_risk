"""
Builds intensity/data/mld_climatology_woa23.nc and strat_climatology_woa23.nc from the World Ocean Atlas 2023
monthly temperature (woa23_decav_tMM_01.nc, https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/temperature/netcdf/decav/1.00/),
on the grid of the original Levitus (1982) maps and with the definitions that reproduce those maps from modern
data (over cyclone-season ocean, 8-30 degrees): mixed layer depth r = 0.95, slope 1.00; stratification r = 0.97, slope 1.05.
    mixed layer: depth at which temperature first falls 1 K below its value at 10 m, or the deepest level with data
                 if it never does (well-mixed shallow seas, polar water); at most 200 m,
    stratification: temperature drop over the 75 m below the mixed layer, K / 100 m, at least 0.1; left out where
                 the column ends first, which the model treats as shallow sea.

Usage: python scripts/build_ocean_maps.py <directory with the WOA23 files> [output directory]
"""
import os
import sys

import numpy as np
import xarray as xr
from scipy.ndimage import map_coordinates

D_T = 1.0
D_Z = 75.0
H_MAX = 200.0

def main(fn_dir, out_dir):
    ds = xr.open_dataset('%s/mld_climatology.nc' % out_dir)
    lat0, lon0_all, month = ds['lat'].values.astype(float), ds['lon'].values.astype(float), ds['month'].values
    ds.close()
    lon0 = lon0_all[:-1]

    T = []
    for m in range(1, 13):
        ds = xr.open_dataset('%s/woa23_decav_t%02d_01.nc' % (fn_dir, m), decode_times = False)
        T.append(ds['t_an'].isel(time = 0).values.astype(float))
        z, lat, lon = ds['depth'].values.astype(float), ds['lat'].values.astype(float), ds['lon'].values.astype(float)
        ds.close()
    T = np.stack(T)                                   # (month, depth, lat, lon)
    k_max = np.searchsorted(z, 800)
    T, z = T[:, :k_max], z[:k_max]
    k10 = np.argmin(np.abs(z - 10))

    def to_old_grid(x):
        o = np.argsort(lon % 360)
        lon2 = (lon % 360)[o]
        x = x[:, o]
        I, J = np.meshgrid((lat0 - lat[0]) / (lat[1] - lat[0]), (lon0 - lon2[0]) / (lon2[1] - lon2[0]), indexing = 'ij')
        a = map_coordinates(np.nan_to_num(x), [I, J], order = 1, mode = 'grid-wrap')
        b = map_coordinates(np.isfinite(x).astype(float), [I, J], order = 1, mode = 'grid-wrap')
        return np.where(b >= 0.5, a / np.maximum(b, 1e-6), np.nan)

    def t_at(h):
        k = np.clip(np.searchsorted(z, np.nan_to_num(h), side = 'right'), 1, len(z) - 1)
        m, a, b = np.indices(h.shape)
        w = (np.nan_to_num(h) - z[k - 1]) / (z[k] - z[k - 1])
        return np.where(np.isfinite(h), T[m, k - 1, a, b] * (1 - w) + T[m, k, a, b] * w, np.nan)

    ref = T[:, k10]
    below = T[:, k10:] <= (ref[:, None] - D_T)
    first = np.argmax(below, axis = 1)
    m, a, b = np.indices(first.shape)
    k = first + k10
    t1, t0 = T[m, k, a, b], T[m, k - 1, a, b]
    frac = np.where(t0 > t1, (t0 - (ref - D_T)) / np.maximum(t0 - t1, 1e-9), 0)
    h = np.where(below.any(axis = 1), z[k - 1] + np.clip(frac, 0, 1) * (z[k] - z[k - 1]), np.nan)
    deepest = np.where(np.isfinite(T), z[None, :, None, None], -1.0).max(axis = 1)
    h = np.where(np.isnan(h) & (deepest >= 0), deepest, h)

    s = 100.0 * (t_at(h) - t_at(h + D_Z)) / D_Z
    s = np.where(np.isfinite(s), np.maximum(s, 0.1), np.nan)
    h = np.minimum(h, H_MAX)

    coords = dict(lat = ('lat', lat0.astype(np.float32)), lon = ('lon', lon0_all.astype(np.float32)), month = ('month', month))
    for name, x, desc, fn in (('mixed_layer', h, 'Climatological mixed-layer depth', 'mld_climatology_woa23.nc'),
                              ('strat', s, 'Climatological sub-mixed layer thermal stratification', 'strat_climatology_woa23.nc')):
        y = np.stack([to_old_grid(x[i]) for i in range(12)], axis = -1)
        y = np.concatenate([y, y[:, :1]], axis = 1)
        ds = xr.Dataset({name: (('lat', 'lon', 'month'), y.astype(np.float32))}, coords = coords,
                        attrs = dict(description = desc, source = 'World Ocean Atlas 2023 temperature, built by scripts/build_ocean_maps.py'))
        ds.to_netcdf('%s/%s' % (out_dir, fn))
        print('Saved %s/%s' % (out_dir, fn))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '%s/../intensity/data' % os.path.dirname(os.path.abspath(__file__)))

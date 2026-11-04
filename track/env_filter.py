"""
Environmental wind filter for the wind statistics that steer and shear the synthetic storms

The daily winds from a reanalysis contain the observed TCs themselves, which inflates the day-to-day 
variability the model uses for the synthetic wind fluctuations, which makes the synthetic TCs move 
too fast and erratically and experience too much random shear. 
Before the monthly statistics are computed, this module 1) removes grid cells within env_wind_tc_radius_km 
of every storm found by util/tc_tracker.py, and 2) smooths each daily field with a Gaussian of width env_wind_sigma_km 
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

import namelist
from util import constants

_tc_cache = {}

""" Returns whether the filter is switched on in the namelist """
def enabled():
    return (getattr(namelist, 'env_wind_sigma_km', 0) > 0) or (getattr(namelist, 'env_wind_tc_radius_km', 0) > 0)

"""
Reads the positions found by tc_tracker, returns (day as datetime64[D], lon, lat) of all TCs to remove
"""
def _read_tc_positions():
    from util import tc_tracker
    fn = tc_tracker.get_fn_positions()
    if fn not in _tc_cache:
        z = np.load(fn)
        _tc_cache[fn] = (z['day'], z['lon'], z['lat'])
    return _tc_cache[fn]

"""
Returns a boolean array (time, lat, lon) that is True within radius_km of any storm position on that day
"""
def tc_mask(days, lat, lon, radius_km):
    tc_day, tc_lon, tc_lat = _read_tc_positions()
    LAT, LON = np.meshgrid(lat, lon % 360.0, indexing = 'ij')
    p2 = np.radians(LAT)
    R = constants.earth_R / 1000.0
    near = np.zeros((len(days), len(lat), len(lon)), bool)
    sel = np.nonzero((tc_day >= days[0]) & (tc_day <= days[-1]))[0]
    for j in sel:
        di = int(np.searchsorted(days, tc_day[j]))        # days need not be contiguous (no-leap calendars)
        if di >= len(days) or days[di] != tc_day[j]:
            continue
        p1 = np.radians(tc_lat[j])
        dl = np.radians(LON - tc_lon[j])
        a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        near[di] |= 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1))) < radius_km
    return near

"""
Normalized Gaussian smoothing on the sphere of x (..., lat, lon), using only cells where valid is True
Latitude first, then longitude with sigma / cos(latitude)
"""
def smooth(x, valid, lat, sigma_km, min_weight = 0.05):
    km_per_deg = np.pi * constants.earth_R / 1000.0 / 180.0
    dlat = abs(float(lat[1] - lat[0]))
    s_lat = sigma_km / km_per_deg / dlat
    w = valid.astype(np.float64)
    xw = np.where(valid, x, 0.0)
    xw = gaussian_filter1d(xw, s_lat, axis = -2, mode = 'nearest')
    w = gaussian_filter1d(w, s_lat, axis = -2, mode = 'nearest')
    out_x = np.empty_like(xw)
    out_w = np.empty_like(w)
    for i, c in enumerate(np.cos(np.radians(lat))):
        s_lon = min(s_lat / max(c, 1e-3), 120.0)
        out_x[..., i, :] = gaussian_filter1d(xw[..., i, :], s_lon, axis = -1, mode = 'wrap')
        out_w[..., i, :] = gaussian_filter1d(w[..., i, :], s_lon, axis = -1, mode = 'wrap')
    return np.where(out_w > min_weight, out_x / np.maximum(out_w, 1e-12), np.nan)

"""
Returns the environmental versions of the daily wind DataArrays ua and va (time, level, lat, lon)
"""
def environmental_winds(ua, va, lat_key, lon_key):
    lat = ua[lat_key].values
    lon = ua[lon_key].values
    t = ua['time'].values
    days = t.astype('datetime64[D]') if np.issubdtype(t.dtype, np.datetime64) else \
        np.array([np.datetime64('%04d-%02d-%02d' % (x.year, x.month, x.day)) for x in t])
    sigma_km = getattr(namelist, 'env_wind_sigma_km', 0)
    radius_km = getattr(namelist, 'env_wind_tc_radius_km', 0)
    valid = np.ones((len(days), len(lat), len(lon)), bool)
    if radius_km > 0:
        valid = ~tc_mask(days, lat, lon, radius_km)
    out = []
    for da in (ua, va):
        x = da.values
        y = np.empty(x.shape, np.float32)
        for k in range(x.shape[1]):       # one level at a time keeps memory low
            if sigma_km > 0:
                y[:, k] = smooth(x[:, k].astype(np.float64), valid, lat, sigma_km)
            else:
                y[:, k] = np.where(valid, x[:, k], np.nan)
        out.append(da.copy(data = y))
    return out[0], out[1]

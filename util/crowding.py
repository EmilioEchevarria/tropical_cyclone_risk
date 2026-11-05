"""
Storm crowding: TCs suppress the formation of others nearby
"""
import numpy as np

import namelist

R_EARTH_KM = 6371.0
DLAT = 2.5
HOURS_PER_MONTH = 730.0
LAT = np.arange(-60, 60.01, DLAT)
LON = np.arange(0, 360, DLAT)

""" Smooths the density (year, month, lat, lon): 0.25, 0.5, 0.25 over months and a flat window of +-half_years years. """
def _smooth(D, half_years):
    out = np.zeros_like(D)
    ny = D.shape[0]
    for yi in range(ny):
        m = D[max(0, yi - half_years):min(ny, yi + half_years + 1)].mean(axis = 0)
        out[yi] = 0.5 * m + 0.25 * (np.roll(m, 1, axis = 0) + np.roll(m, -1, axis = 0))
    return out

""" Expected suppression C at each formation point, for the smoothed density D. """
def _crowd_index(D, g_year, g_month, g_lon, g_lat):
    C = np.zeros(g_lon.size)
    phi, lam = np.radians(g_lat), np.radians(g_lon)
    sin_c, cos_c = np.sin(np.radians(LAT)), np.cos(np.radians(LAT))
    lon_c = np.radians(LON)
    for yi in range(D.shape[0]):
        for m in range(12):
            q = np.nonzero((g_year == yi) & (g_month == m))[0]
            a, o = np.nonzero(D[yi, m] > 0)
            if (len(q) == 0) or (len(a) == 0):
                continue
            cd = np.outer(np.sin(phi[q]), sin_c[a]) + np.outer(np.cos(phi[q]), cos_c[a]) * np.cos(lam[q][:, None] - lon_c[o][None, :])
            dist = R_EARTH_KM * np.arccos(np.clip(cd, -1, 1))
            C[q] = (np.interp(dist, namelist.crowd_km, namelist.crowd_suppress, right = 0.) * D[yi, m, a, o][None, :]).sum(axis = 1)
    return C

"""
Chance each track survives crowding. lon, lat, vmax are (track, time) with vmax in m/s, output_step_hours apart; year and
month (1-12) per track; years and seeds_per_year give the seeds drawn in each year. The storms per year represented by a
track are K / seeds_per_year: with constant K as given, otherwise K such that the kept storms make storms_per_year.
Returns (chance kept, K). Tracks that never reach tropical-storm strength get the mean chance. Each storm counts in its own
density, which is negligible at a hundred tracks per year over several years but not in a short run of a few tracks.
"""
def keep_probability(lon, lat, vmax, year, month, years, seeds_per_year, K = None, storms_per_year = None,
                     output_step_hours = 1, half_years = 2):
    step = max(1, int(round(6. / output_step_hours)))
    lon6, lat6, v6 = lon[:, ::step], lat[:, ::step], vmax[:, ::step]
    ts = np.isfinite(lon6) & np.isfinite(v6) & (v6 >= namelist.seed_vmax_threshold_ms) & (np.abs(lat6) < LAT[-1])
    has = ts.any(axis = 1)
    first = np.argmax(ts, axis = 1)
    y_i = np.searchsorted(years, year)
    m_i = (np.asarray(month).astype(int) - 1) % 12
    s, t = np.nonzero(ts)
    a = np.round((lat6[s, t] - LAT[0]) / DLAT).astype(int)
    o = np.round((lon6[s, t] % 360.) / DLAT).astype(int) % LON.size
    g = np.nonzero(has)[0]
    g_lon, g_lat = lon6[g, first[g]] % 360., lat6[g, first[g]]
    per_track = 1. / np.asarray(seeds_per_year)[y_i]
    ny = len(years)
    P = np.ones(lon.shape[0])
    k = K if K is not None else storms_per_year / np.sum(per_track[has] / ny)
    for _ in range(6):
        D = np.zeros((ny, 12, LAT.size, LON.size))
        np.add.at(D, (y_i[s], m_i[s], a, o), (k * per_track * P)[s] * 6. / HOURS_PER_MONTH)
        C = _crowd_index(_smooth(D, half_years), y_i[g], m_i[g], g_lon, g_lat)
        P[g] = np.exp(-C)
        if K is None:
            k = storms_per_year / np.sum((P * per_track)[has] / ny)
    P[~has] = P[g].mean() if len(g) else 1.
    return P, k

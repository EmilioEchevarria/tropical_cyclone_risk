#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jzlin@mit.edu
"""
# %%
import datetime
import numpy as np
import xarray as xr
import time

from scipy.interpolate import interp1d

import namelist
from track import env_wind
from util import input, mat, sphere

"""
Generate F from Emanuel et. al. (2006). It is a Fourier series where
the individual wave components have a random phase. In addition, the
kinetic energy power spectrum follows that of geostrophic turublence.
"""
def gen_f(N, T, t, num):
    fs = np.zeros((num, np.size(t)))
    p = getattr(namelist, 'fluct_spectrum_exponent', 1.5)
    for i in range(0, num):
        n = np.linspace(1, N, N)
        xln = np.tile(np.random.rand(N, 1), (1, np.size(t)))   # Zero phase correlation
        fs[i, :] = np.sqrt(2 / np.sum(np.power(n, -2 * p))) * \
                   np.sum(np.multiply(np.tile(np.power(n, -p), (np.size(t), 1)).T,
                                      np.sin(2. * np.pi * (np.outer(n, t) / T + xln))), axis=0)
    return(fs)

"""
Drift of a storm at (clon, clat) that the monthly environmental flow does not explain
Beta drift is part of it, but not all: the zonal part differs between regions by more than 2 m/s, the same at
the weak stage and the storm stage (monsoon-trough regions, where parent disturbances hardly propagate, against trade-wind
regions, where they travel west), and fades towards the westerlies. The poleward part is flat at low latitudes and grows
beyond drift_lat0, where storms recurve around the subtropical ridge.
"""
def drift_speed(clon, clat):
    lats = np.abs(clat)
    nodes, values = namelist.u_drift_nh if clat >= 0 else namelist.u_drift_sh
    u = np.interp(clon % 360., nodes, values, period = 360.)
    u *= np.clip((namelist.u_drift_lat1 - lats) / (namelist.u_drift_lat1 - namelist.u_drift_lat0), 0., 1.)
    slope = namelist.v_beta_slope if clat >= 0 else namelist.v_beta_slope_sh
    w = namelist.v_beta + slope * max(0., lats - namelist.drift_lat0) / 30.
    w += getattr(namelist, 'v_beta_low', 0.) * np.clip((namelist.v_beta_low_lat1 - lats) / (namelist.v_beta_low_lat1 - namelist.v_beta_low_lat0), 0., 1.)
    regional = getattr(namelist, 'v_drift_nh' if clat >= 0 else 'v_drift_sh', None)
    if regional is not None:
        w += np.interp(clon % 360., regional[0], regional[1], period = 360.) * np.clip((namelist.u_drift_lat1 - lats) / (namelist.u_drift_lat1 - namelist.u_drift_lat0), 0., 1.)
    # Drift never carries a storm towards the equator
    return (u, np.sign(clat) * max(w, 0.))

"""
Seed the generator. Advantage of this method is that processes that
run close to each other will have very different seeds.
"""
def random_seed():
    t = int(time.time() * 1000.0)
    np.random.seed(((t & 0xff000000) >> 24) +
                   ((t & 0x00ff0000) >>  8) +
                   ((t & 0x0000ff00) <<  8) +
                   ((t & 0x000000ff) << 24))

class BetaAdvectionTrack:
    """
    Class that defines methods to generate synthetic tracks using a simple
    beta-advection model.
    """
    def __init__(self, fn_wnd_stat, basin, dt_start, dt_track = 3600,
                 total_time = 15*24*60*60):
        self.fn_wnd_stat = fn_wnd_stat
        self.dt_track = dt_track                # numerical time step (seconds)
        self.total_time = total_time            # total time of track (seconds)
        self.total_steps = int(self.total_time / self.dt_track) + 1
        self.t_s = np.linspace(0, self.total_time, int(self.total_time / self.dt_track) + 1)
        self.T_Fs = namelist.T_days*24*60*60    # period of the fourier series
        # Fraction of the synthetic wind fluctuation that steers the storm
        self.steer_fluct_scale = getattr(namelist, 'steer_fluct_scale', 1.0)
        self.nLvl = len(namelist.steering_levels)
        self.nWLvl = self.nLvl * 2
        self.dt_start = dt_start.astype('datetime64[ns]')
        self.basin = basin
        self.var_names = env_wind.wind_mean_vector_names()
        self.u_Mean_idxs = np.zeros(self.nLvl).astype(int)
        self.v_Mean_idxs = np.zeros(self.nLvl).astype(int)
        p_lvls = namelist.steering_levels
        for i in range(self.nLvl):
            self.u_Mean_idxs[i] = int(self.var_names.index('ua' + str(p_lvls[i]) + '_Mean'))
            self.v_Mean_idxs[i] = int(self.var_names.index('va' + str(p_lvls[i]) + '_Mean'))          
        self._load_wnd_stat()

    def _interp_basin_field(self, var):
        lon_b, lat_b, var_b = self.basin.transform_global_field(self.wnd_lon, self.wnd_lat, var)
        return mat.interp2_fx(lon_b, lat_b, np.nan_to_num(var_b))

    def _load_wnd_stat(self):
        wnd_Mean, wnd_Cov = env_wind.read_env_wnd_fn(self.fn_wnd_stat)
        self.wnd_Mean_Fxs = [0]*len(wnd_Mean)
        self.wnd_Cov_Fxs = [['' for i in range(len(wnd_Cov))] for j in range(len(wnd_Cov[0]))]
        ds = xr.open_dataset(self.fn_wnd_stat)
        self.datetime_start = input.convert_to_datetime(ds, np.array([self.dt_start]))
        self.wnd_lon = wnd_Mean[0]['lon']
        self.wnd_lat = wnd_Mean[0]['lat']

        # Since xarray interpolation is slow, use our own 2-D interpolation.
        # Only create interpolation functions for the lower trianglular matrix.
        for i in range(len(wnd_Mean)):
            self.wnd_Mean_Fxs[i] = self._interp_basin_field(wnd_Mean[i].interp(time = self.dt_start))
            for j in range(len(wnd_Mean)):
                if j <= i:
                    self.wnd_Cov_Fxs[i][j] = self._interp_basin_field(wnd_Cov[i][j].interp(time = self.dt_start))

    def interp_wnd_mean_cov(self, clon, clat, ct):
        wnd_mean = np.zeros(self.nWLvl)
        wnd_cov = np.zeros((self.nWLvl, self.nWLvl))

        # Only interpolate the lower trianglular matrix.
        for i in range(0, self.nWLvl):
            wnd_mean[i] = self.wnd_Mean_Fxs[i].ev(clon, clat)
            for j in range(0, self.nWLvl):
                if j <= i:
                    wnd_cov[i, j] = self.wnd_Cov_Fxs[i][j].ev(clon, clat)

        for i in range(0, self.nWLvl):
            for j in range(i, self.nWLvl):
                wnd_cov[i, j] = wnd_cov[j, i]

        return(wnd_mean, wnd_cov)

    """ Generate the random Fourier Series """
    def gen_synthetic_f(self):
        N_series = 15                       # number of sine waves
        return(gen_f(N_series, self.T_Fs, self.t_s, self.nWLvl))

    """ Calculate environmental winds at a point and time.
    With parts = True, returns (winds, mean winds, fluctuation) instead """
    def _env_winds(self, clon, clat, ts, parts = False):
        if np.isnan(clon) or np.isnan(ts):
            zeros = np.zeros(self.nWLvl)
            return (zeros, zeros, zeros) if parts else zeros

        ct = self.datetime_start + datetime.timedelta(seconds = ts)
        wnd_mean, wnd_cov = self.interp_wnd_mean_cov(clon, clat, ct)
        try:
            wnd_A = np.linalg.cholesky(wnd_cov)
            wnd_fluct = np.matmul(wnd_A, self.Fs_i(ts))
        except np.linalg.LinAlgError as err:
            # Covariance is not positive definite here (rare): use the mean wind
            wnd_fluct = np.zeros(self.nWLvl)
        wnds = wnd_mean + wnd_fluct
        return (wnds, wnd_mean, wnd_fluct) if parts else wnds

    """ Calculate the translational speeds from the beta advection model """
    def _step_bam_track(self, clon, clat, ts, steering_coefs):
        # Include a hard stop for latitudes above 80 degrees.
        # Ensures that solve_ivp does not go past the domain bounds.
        if np.abs(clat) >= 80:
            return (np.zeros(2), np.zeros(self.nWLvl))
        wnds, wnd_mean, wnd_fluct = self._env_winds(clon, clat, ts, parts = True)
        fluct = self.steer_fluct_scale * wnd_fluct
        steer = wnd_mean + fluct

        v_bam = np.zeros(2)
        u_drift, v_drift = drift_speed(clon, clat)

        v_bam[0] = np.dot(steer[self.u_Mean_idxs], steering_coefs) + u_drift
        v_bam[1] = np.dot(steer[self.v_Mean_idxs], steering_coefs) + v_drift
        return(v_bam, wnds)

    """ Calculate the steering coefficients. """
    def _calc_steering_coefs(self):
        assert len(namelist.steering_coefs) == len(namelist.steering_levels)
        steering_coefs = np.array(namelist.steering_coefs)
        return steering_coefs

    """ Generate a track with a starting position of (clon, clat) """
    def gen_track(self, clon, clat):
        # Make sure that tracks are sufficiently randomized.
        random_seed()

        # Create the weights for the beta-advection model (across time).
        self.Fs = self.gen_synthetic_f()
        self.Fs_i = interp1d(self.t_s, self.Fs, axis = 1)

        track = np.full((self.total_steps+1, 2), np.nan)
        wind_track = np.full((self.total_steps, self.nWLvl), np.nan)
        v_trans_track = np.full((self.total_steps, 2), np.nan)

        track[0, 0] = clon; track[0, 1] = clat;
        lonC, latC = clon, clat
        for ts in range(0, self.total_steps):
            (v_bam, wind_track[ts, :]) = self._step_bam_track(lonC, latC, ts, self._calc_steering_coefs())
            v_trans_track[ts] = v_bam
            dx = v_bam[0] * self.dt_track
            dy = v_bam[1] * self.dt_track
            (lonC, latC) = sphere.to_sphere_dist(lonC, latC, dx, dy)
            track[ts+1, 0] = lonC; track[ts+1, 1] = latC;

            if not self.basin.in_basin(lonC, latC, 1):
                break

        return (track[:-1, :], v_trans_track, wind_track)

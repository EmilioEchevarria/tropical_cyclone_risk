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
    n = np.arange(1, N + 1)
    amp = np.sqrt(2.0 / np.sum(np.power(n, -3.0))) * np.power(n, -1.5)
    sin_base = 2.0 * np.pi * np.outer(n, t) / T

    for i in range(0, num):
        # n = np.linspace(1, N, N)
        # xln = np.tile(np.random.rand(N, 1), (1, np.size(t)))   # Zero phase correlation
        # fs[i, :] = np.sqrt(2 / np.sum(np.power(n, -3))) * \
        #            np.sum(np.multiply(np.tile(np.power(n, -1.5), (np.size(t), 1)).T,
        #                               np.sin(2. * np.pi * (np.outer(n, t) / T + xln))), axis=0)
        phases = np.random.rand(N, 1)       # zero phase correlation
        fs[i, :] = amp @ np.sin(sin_base + 2.0 * np.pi * phases)
    return(fs)

"""
Same as gen_f but takes the precomputed (amp, sin_base) arrays, avoids 
rebuilding the outer product and the N-power vector on every call
"""
def _gen_f_precomputed(amp, sin_base, num):
    N = amp.shape[0]
    fs = np.empty((num, sin_base.shape[1]))
    for i in range(num):
        phases = np.random.rand(N, 1)
        fs[i, :] = amp @ np.sin(sin_base + 2.0 * np.pi * phases)
    return fs

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
        self.T_Fs = namelist.T_days*24*60*60    # 15-day period of the fourier series
        self.u_beta = namelist.u_beta           # zonal beta drift speed
        self.v_beta = namelist.v_beta           # meridional beta drift speed
        self.nLvl = len(namelist.steering_levels)
        self.nWLvl = self.nLvl * 2
        self.dt_start = dt_start.astype('datetime64[ns]')
        self.basin = basin
        self.var_names = env_wind.steering_wind_mean_names()
        self.u_Mean_idxs = np.zeros(self.nLvl).astype(int)
        self.v_Mean_idxs = np.zeros(self.nLvl).astype(int)
        p_lvls = namelist.steering_levels
        for i in range(self.nLvl):
            self.u_Mean_idxs[i] = int(self.var_names.index('ua' + str(p_lvls[i]) + '_Mean'))
            self.v_Mean_idxs[i] = int(self.var_names.index('va' + str(p_lvls[i]) + '_Mean'))
        
        # Precompute the Fourier-series coefficients for gen_synthetic_f. N, T, and t_s never change 
        # across tracks, so amp and sin_base are constants and only the random phases vary per track
        _N_series = 15
        _n_arr = np.arange(1, _N_series + 1)
        self._gen_f_N = _N_series
        self._gen_f_amp = (np.sqrt(2.0 / np.sum(np.power(_n_arr, -3.0))) * np.power(_n_arr, -1.5))
        self._gen_f_sin_base = 2.0 * np.pi * np.outer(_n_arr, self.t_s) / self.T_Fs

        # Pre-allocate buffers used on the hot path (_env_winds)
        self._wnd_mean_buf = np.zeros(self.nWLvl)
        self._wnd_cov_buf = np.zeros((self.nWLvl, self.nWLvl))

        # Constants used by the custom linear interpolator for self.Fs
        self._Fs_inv_dt = 1.0 / self.dt_track
        self._Fs_nmax = self.total_steps - 1

        # Precompute the Fourier-series coefficients for the high-frequency wiggle:
        self._wiggle_amp_ms = float(getattr(namelist, 'wiggle_amp_ms', 0.0))
        self._wiggle_v_scale_ms = float(getattr(namelist, 'wiggle_v_scale_ms', 20.0))
        self._wiggle_enabled = self._wiggle_amp_ms > 0.0
        if self._wiggle_enabled:
            _Nw = int(getattr(namelist, 'N_wiggle', 5))
            _Tw = float(getattr(namelist, 'T_wiggle_days', 1.0)) * 24.0 * 3600.0
            _nw_arr = np.arange(1, _Nw + 1)
            self._wiggle_N = _Nw
            self._wiggle_amp_coefs = (np.sqrt(2.0 / np.sum(np.power(_nw_arr, -3.0))) * np.power(_nw_arr, -1.5))
            self._wiggle_sin_base = 2.0 * np.pi * np.outer(_nw_arr, self.t_s) / _Tw

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
            self.wnd_Mean_Fxs[i] = self._interp_basin_field(wnd_Mean[i].sel(time = self.dt_start, method='nearest'))
            for j in range(len(wnd_Mean)):
                if j <= i:
                    self.wnd_Cov_Fxs[i][j] = self._interp_basin_field(wnd_Cov[i][j].sel(time = self.dt_start, method='nearest'))

    def interp_wnd_mean_cov(self, clon, clat, ct):
        #wnd_mean = np.zeros(self.nWLvl)
        #wnd_cov = np.zeros((self.nWLvl, self.nWLvl))
        
        # Reuse pre-allocated buffers, called on every ODE step
        wnd_mean = self._wnd_mean_buf
        wnd_cov = self._wnd_cov_buf
        nW = self.nWLvl

        # # Only interpolate the lower trianglular matrix.
        # for i in range(0, self.nWLvl):
        #     wnd_mean[i] = self.wnd_Mean_Fxs[i].ev(clon, clat)
        #     for j in range(0, self.nWLvl):
        #         if j <= i:
        #             wnd_cov[i, j] = self.wnd_Cov_Fxs[i][j].ev(clon, clat)

        # for i in range(0, self.nWLvl):
        #     for j in range(i, self.nWLvl):
        #         wnd_cov[i, j] = wnd_cov[j, i]

        # Only interpolate the lower-triangular part of the covariance. np.linalg.cholesky 
        # reads only the lower triangle, so we don't need to mirror to the upper
        wnd_Mean_Fxs = self.wnd_Mean_Fxs
        wnd_Cov_Fxs = self.wnd_Cov_Fxs
        for i in range(nW):
            wnd_mean[i] = wnd_Mean_Fxs[i].ev(clon, clat)
            row = wnd_Cov_Fxs[i]
            for j in range(i + 1):
                wnd_cov[i, j] = row[j].ev(clon, clat)

        return(wnd_mean, wnd_cov)

    """ Generate the random Fourier Series """
    def gen_synthetic_f(self):
        #N_series = 15                       # number of sine waves
        #return(gen_f(N_series, self.T_Fs, self.t_s, self.nWLvl))
        return _gen_f_precomputed(self._gen_f_amp, self._gen_f_sin_base, self.nWLvl)

    """ Fast linear interpolation of self.Fs at time ts (seconds), rquivalent to scipy.interpolate.interp1d """
    def _fs_at(self, ts):
        idx = ts * self._Fs_inv_dt
        lo = int(idx)
        if lo < 0:
            return self.Fs[:, 0]
        if lo >= self._Fs_nmax:
            return self.Fs[:, self._Fs_nmax]
        frac = idx - lo
        return self.Fs[:, lo] * (1.0 - frac) + self.Fs[:, lo + 1] * frac

    """ Generate the high-frequency "wiggle" Fourier series """
    def gen_wiggle_f(self):
        return _gen_f_precomputed(self._wiggle_amp_coefs, self._wiggle_sin_base, 2)

    """ Fast uniform-grid linear interpolation for the wiggle series """
    def _fs_wiggle_at(self, ts):
        idx = ts * self._Fs_inv_dt
        lo = int(idx)
        if lo < 0:
            return self.Fs_wiggle[:, 0]
        if lo >= self._Fs_nmax:
            return self.Fs_wiggle[:, self._Fs_nmax]
        frac = idx - lo
        return self.Fs_wiggle[:, lo] * (1.0 - frac) + self.Fs_wiggle[:, lo + 1] * frac

    """ Calculate environmental winds at a point and time. """
    def _env_winds(self, clon, clat, ts):
        if np.isnan(clon) or np.isnan(ts):
            return np.zeros(self.nWLvl)

        #ct = self.datetime_start + datetime.timedelta(seconds = ts)
        wnd_mean, wnd_cov = self.interp_wnd_mean_cov(clon, clat, ts)
        try:
            wnd_A = np.linalg.cholesky(wnd_cov)
        except np.linalg.LinAlgError as err:
            print(self.dt_start)
            return np.zeros(self.nWLvl)
        #wnds = wnd_mean + np.matmul(wnd_A, self.Fs_i(ts))
        wnds = wnd_mean + wnd_A @ self._fs_at(ts)
        return wnds

    """ Calculate the translational speeds from the beta advection model. 
        If v is given and the wiggle is enabled, adds an intensity-modulated
        high-freq perturbation of amplitude wiggle_amp_ms*exp(-v/wiggle_v_scale_ms) """
    def _step_bam_track(self, clon, clat, ts, steering_coefs, v=None):
        # Include a hard stop for latitudes above 80 degrees.
        # Ensures that solve_ivp does not go past the domain bounds.
        if np.abs(clat) >= 80:
            return (np.zeros(2), np.zeros(self.nWLvl))
        wnds = self._env_winds(clon, clat, ts)

        v_bam = np.zeros(2)
        w_lat = np.cos(np.deg2rad(clat))
        v_beta_sgn = np.sign(clat) * self.v_beta

        v_bam[0] = np.dot(wnds[self.u_Mean_idxs], steering_coefs) + self.u_beta * w_lat
        v_bam[1] = np.dot(wnds[self.v_Mean_idxs], steering_coefs) + v_beta_sgn * w_lat

        # Intensity-dependent wiggle
        if self._wiggle_enabled and v is not None and hasattr(self, 'Fs_wiggle'):
            sigma = self._wiggle_amp_ms * np.exp(-max(v, 0.0) / self._wiggle_v_scale_ms)
            if sigma > 0:
                eta = self._fs_wiggle_at(ts)
                v_bam[0] += sigma * eta[0]
                v_bam[1] += sigma * eta[1]

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
        #self.Fs_i = interp1d(self.t_s, self.Fs, axis = 1)
        if self._wiggle_enabled:
            self.Fs_wiggle = self.gen_wiggle_f()

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

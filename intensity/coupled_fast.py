#!/usr/bin/env python
"""
Author: Jonathan Lin
Implementation of FAST (Emanuel, 2017) that is coupled to the track.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, RectBivariateSpline
import warnings

import namelist
from intensity import geo
from thermo import thermo
from track import bam_track, env_wind
from util import constants

class Coupled_FAST(bam_track.BetaAdvectionTrack):
    def __init__(self, fn_wnd_stat, basin, dt_start, dt_s, total_time_s):
        super().__init__(fn_wnd_stat, basin, dt_start, dt_s, total_time_s)

        """ FAST Constants """
        self.Ck = namelist.Ck                       # surface enthalpy coefficient
        self.h_bl = namelist.atm_bl_depth           # boundary layer depth (m)
        self.epsilon = 0.33                         # thermodynamic efficiency
        self.kappa = 0.1                            # dimensionless parameter
        self.beta = 1 - self.epsilon - self.kappa   # dimensionless parameter

        # Read in high-resolution bathymetry and land masks.
        self.f_bath = geo.read_bathy(basin)
        self.f_land = geo.read_land(basin)
        self.debug = False

        # Variable drag coefficient
        self.f_Cd = geo.read_drag(basin)

    """ Return if over land (True) or ocean (False) """
    def _get_over_land(self, clon, clat):
        # 9/2/2020: Changed to 1 (and not a rounded number), since PI
        # is changed to be reduced by the area of the core over land.
        return(self.f_land.ev(clon, clat).flatten()[0] == 1)

    """ Return the current bathymetry/topography at a position."""
    def _get_current_bathymetry(self, clon, clat):
        return self.f_bath.ev(clon, clat).flatten()[0]

    """ Return the current mixed layer depth at a position. """
    def _get_current_mixed(self, clon, clat):
        return self.f_mld.ev(clon, clat).flatten()[0]

    """ Return the current drag coefficient at a position. """
    def _get_current_Cd(self, clon, clat):
        # As in Emanuel (2017), the drag rises at landfall: a centre still over the sea keeps the over-ocean value.
        if getattr(namelist, 'land_drag_on_land_only', False) and (self.f_land.ev(clon, clat).flatten()[0] < 0.5):
            return namelist.Cd
        return (self.f_Cd.ev(clon, clat).flatten()[0])

    """ Return the current sub-mixed layer thermal stratification, (K / 100 m)
        at time t_s."""
    def _get_current_strat(self, clon, clat):
        return self.f_strat.ev(clon, clat).flatten()[0]

    """ Return the current potential intensity at a position"""
    def _get_current_vpot(self, clon, clat):
        if self._get_over_land(clon, clat):
            return(0)
        else:
            return self.f_vpot.ev(clon, clat).flatten()[0]

    """ Calculate, alpha (Equation 4), the ocean feedback parameter.
    Ocean mixing is turned off when the mixed layer depth equals or
    exceeds the local ocean depth (alpha = 1), or when the hurricane
    is over land.
    """
    def _calc_alpha(self, clon, clat, v_trans, v):
        h_m = self._get_current_mixed(clon, clat)
        t_strat = self._get_current_strat(clon, clat)
        v_pot = self._get_current_vpot(clon, clat)
        u_T = np.linalg.norm(v_trans)
        bathymetry = self._get_current_bathymetry(clon, clat)

        # When over land, h_m = NaN, so we make alpha = 1.
        if (bathymetry >= 0):
            return(1)
        elif (-h_m <= bathymetry or t_strat == 0):
            return self._alpha_shallow(v, u_T, -bathymetry if h_m <= 0 else min(h_m, -bathymetry))
        else:
            z = self._calc_z(v, h_m, v_pot, u_T, t_strat) * self._ocean_f_factor(clat)
            try:
                fac = np.exp(-np.clip(z, 0, 100))
            except OverflowError:
                fac = np.nan
            return(1 - 0.87 * fac)

    """ Ocean feedback where no colder water can be mixed up (the mixed layer reaches the sea floor): the surface enthalpy flux
    still cools the whole column of depth D (m) while the storm passes over a width 2 r, with r = shelf_radius_km. Cooling dT = F (2 r / u_T) /
    (rho_w c_w D) with F = rho_a Ck v dk takes dk*/dT dT from the air-sea enthalpy difference dk, so alpha = 1 - dk*/dT dT / dk, in which dk cancels. """
    def _alpha_shallow(self, v, u_T, depth):
        r = getattr(namelist, 'shelf_radius_km', 0.0) * 1000.
        if (r <= 0) or (depth <= 0):
            return(1)
        rho_a, rho_w, c_w, dkdT = 1.15, 1025., 4000., 4800.
        cool = rho_a * self.Ck * max(v, 0.) * (2 * r / max(u_T, 0.5)) * dkdT / (rho_w * c_w * depth)
        return(max(1 - cool, 0.13))

    """ Latitude dependence of the ocean feedback in the fit of Schade and Emanuel (1999), z ~ f^0.59 
    Where the Earth's rotation is weaker, near-inertial currents grow larger and mix up more cold water. 
    Held below 5 degrees latitude, outside the range the fit explored. """
    def _ocean_f_factor(self, clat):
        exponent = getattr(namelist, 'ocean_f_exponent', 0.0)
        if exponent == 0:
            return 1.0
        f = 2 * 7.292e-5 * max(abs(np.sin(np.deg2rad(clat))), np.sin(np.deg2rad(5.)))
        return (f / 5e-5) ** exponent

    """ Calculate z (Equation 5) at a time t.
    v is the current intensity in m/s.
    h_m is the ocean mixed layer depth in meters.
    v_pot is the potential intensity in m/s.
    u_T is the storm translation speed in m/s.
    t_strat is the sub-mixed layer thermal stratification in K per 100 m.
    """
    def _calc_z(self, v, h_m, v_pot, u_T, t_strat):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            z = 0.01 * (t_strat ** -0.4) * h_m * u_T * v_pot / v
        return z

    """ Calculate the dimensionless parameter beta (Equation 6). """
    def _calc_beta(self):
        return self.beta

    """ Calcuate the dimensionless parameter gamma (Equation 7).
        gamma varies with v, since it is a function of alpha."""
    def _calc_gamma(self, alpha):
        return self._calc_epsilon() + alpha * self._calc_kappa()

    """ Calculate the dimensionless parameter epsilon (Equation 8). """
    def _calc_epsilon(self):
        return self.epsilon

    """ Calculate the dimensionless parameter kappa (Equation 9). """
    def _calc_kappa(self):
        return self.kappa

    """ Calculate the shear vector from a vector of environmental winds.
        The shear vector is [zonal shear, meridional shear]."""
    def _calc_env_wnd_to_shr(self, env_wnds):
        u250, v250, u850, v850 = env_wind.deep_layer_winds(np.expand_dims(env_wnds, axis = 0))
        shr = np.array([u250 - u850, v250 - v850]).flatten()
        return shr

    """ Calculate the magnitude of the 250-850 hPa environmental wind shear."""
    def _calc_S(self, env_wnds):
        return np.linalg.norm(self._calc_env_wnd_to_shr(env_wnds))

    """ Calculate the normalized mid-level saturation entropy deficit."""
    def _calc_chi(self, clon, clat):
        return self.f_chi.ev(clon, clat).flatten()[0]

    """ Boundary-layer depth (m) for a vortex of intensity v (m/s) at latitude clat. """
    def _h_bl(self, v, clat):
        if not getattr(namelist, 'bl_inertial_scaling', False):
            return self.h_bl
        v_ref = namelist.bl_v_ref_ms
        if not (v > v_ref):
            return self.h_bl
        f = 2 * 7.292e-5 * abs(np.sin(np.deg2rad(clat)))
        r = namelist.bl_rmw_km * 1000.
        inertial = lambda u: np.sqrt((f + 2 * u / r) * (f + u / r))
        return self.h_bl * np.sqrt(inertial(v_ref) / inertial(v))

    """ Calculate the ventilation. The vortex also feels shear on scales the smoothed environmental winds leave out. """
    def _calc_venti(self, t, clon, clat, env_wnds):
        chi = self._calc_chi(clon, clat)
        shr = self._calc_env_wnd_to_shr(env_wnds)
        if self.shear_noise is not None:
            shr = shr + np.array([np.interp(t, self.noise_t_s, self.shear_noise[0]),
                                  np.interp(t, self.noise_t_s, self.shear_noise[1])])
        return (np.linalg.norm(shr) * chi)

    """ Unit-variance red noise with e-folding time tau_hours, on the grid noise_t_s. """
    def _red_noise(self, tau_hours):
        r = np.exp(-(self.noise_t_s[1] - self.noise_t_s[0]) / (tau_hours * 3600.))
        x = np.random.standard_normal(self.noise_t_s.size)
        x[1:] *= np.sqrt(1 - r ** 2)
        for i in range(1, x.size):
            x[i] += r * x[i - 1]
        return x

    """ Draw the fluctuations a track feels: the shear on scales the environmental winds leave out (zonal, meridional,
    m/s) and the fluctuating supply of inner-core moisture (see _dmdt). """
    def _gen_noise(self):
        self.noise_t_s = np.arange(0, self.total_time + 3 * 3600., 3 * 3600.)
        sigma_s = getattr(namelist, 'shear_subscale_ms', 0.0)
        sigma_m = getattr(namelist, 'moisture_noise', 0.0)
        self.shear_noise = None if sigma_s == 0 else \
            sigma_s * np.array([self._red_noise(namelist.shear_subscale_tau_hours) for _ in range(2)])
        self.moisture_noise = None if sigma_m == 0 else sigma_m * self._red_noise(namelist.moisture_noise_tau_hours)

    """ Ventilation of a vortex of intensity v (m/s) at latitude clat, relative to a vortex of bl_v_ref_ms at
    f = 5e-5 /s. Shear and the dry air it brings in matter less the stiffer the circulation is, ventilation ~ I^(-1/2)
    (the Ekman scaling of the inflow layer that downdrafts flush), with I = sqrt((f + 2u/r)(f + u/r)) at r = vent_radius_km,
    outside the eyewall, where u = v (bl_rmw_km / r)^(1/2). Vortices weaker than bl_v_ref_ms feel only the change in f,
    which is held below 3 degrees latitude. """
    def _vent_factor(self, v, clat):
        exponent = getattr(namelist, 'vent_inertial_exponent', 0.0)
        if exponent == 0:
            return 1.0
        r = namelist.vent_radius_km * 1000.
        shape = np.sqrt(namelist.bl_rmw_km / namelist.vent_radius_km)
        inertial = lambda u, f: np.sqrt((f + 2 * shape * u / r) * (f + shape * u / r))
        f = 2 * 7.292e-5 * max(abs(np.sin(np.deg2rad(clat))), np.sin(np.deg2rad(3.)))
        v_ref = namelist.bl_v_ref_ms
        return (inertial(v_ref, 5e-5) / inertial(max(v, v_ref), f)) ** exponent

    """ Define the first ODE in the coupled set, Equation 2.
    For now, use a drag coefficient, constant boundary layer depth,
    constant thermodynamic efficiency (Equation 8), and constant
    kappa (Equation 9). This means beta is constant (Equation 6).
    However, alpha still varies (Equation 4, 5) with time, which means
    gamma must vary with time as well (Equation 7).
    Potential intensity (v_p) also varies with time.
    """
    def _dvdt(self, clon, clat, v, m, v_trans, env_wnds, t, u_ocean = None):
        v_pot = self._get_current_vpot(clon, clat)
        alpha = self._calc_alpha(clon, clat, v_trans if u_ocean is None else u_ocean, v)
        gamma = self._calc_gamma(alpha)
        beta = self._calc_beta()
        # In Emanuel (2012), Emanuel (2017), there are differing expressions for
        # the intensification rate. It seems Ck and Cd are used interchangeably.
        # Here, in lieu of Ck, we use Cd, and include a variable drag coefficient 
        # to represent faster decay over land.
        Cd = self._get_current_Cd(clon, clat)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if getattr(namelist, 'land_drag_friction_only', False):
                # Rough land raises friction, not the energy supply, so the growth term keeps the over-ocean coefficient. On rough
                # cells next to the sea potential intensity is still full, and a growth term scaled by the land's drag gains 20 kt in an hour.
                dvdt = (0.5 / self._h_bl(v, clat) * (namelist.Cd * alpha * beta * (v_pot ** 2) * (m ** 3) -
                                                     Cd * (1 - gamma * (m ** 3)) * (v ** 2)))
            else:
                dvdt = (0.5 * Cd / self._h_bl(v, clat) * (alpha * beta * (v_pot ** 2) * (m ** 3) -
                                                     (1 - gamma * (m ** 3)) * (v ** 2)))
        # Over water, friction spins the vortex down faster than the rate h_bl gives, which is set by spin-up: only part of
        # the convective heating goes into spinning the vortex up. Land keeps its own drag.
        if (dvdt < 0) and not self._get_over_land(clon, clat):
            dvdt = dvdt * getattr(namelist, 'spindown_fac', 1.0)
        return dvdt if ~np.isnan(dvdt) else 0

    """ Initializes m if no m has been given. Assumes an initial dvdt. """
    def _init_m(self, y, dvdt):
        steering_coefs = self._calc_steering_coefs(y[2])
        v_bam, env_wnds = self._step_bam_track(y[0], y[1], 0., steering_coefs, y[2])
        v_pot = np.max([self._get_current_vpot(y[0], y[1]),
                        self._get_current_vpot(y[0]-0.25, y[1]-0.25),
                        self._get_current_vpot(y[0]-0.25, y[1]+0.25),
                        self._get_current_vpot(y[0]+0.25, y[1]-0.25),
                        self._get_current_vpot(y[0]+0.25, y[1]+0.25)])
        alpha = self._calc_alpha(y[0], y[1], v_bam, y[2])
        gamma = self._calc_gamma(alpha)
        beta = self._calc_beta()

        Cd = self._get_current_Cd(y[0], y[1])
        numer = 2 * self._h_bl(y[2], y[1]) / Cd * dvdt + (y[2] ** 2)
        denom = alpha * beta * (v_pot ** 2) + gamma * (y[2] ** 2)
        return(np.maximum(np.minimum(np.cbrt(numer / denom), 1), 0))

    """ Define the second ODE in the coupled set, Equation 3.
    For now, use a drag coefficient, constant boundary layer depth,
    constant thermodynamic efficiency (Equatton 8), and constant
    kappa (Equation 9).
    However, environmental wind shear (S) still varies with time.
    """
    def _dmdt(self,  clon, clat, v, m, env_wnds, t):
        venti = self._calc_venti(t, clon, clat, env_wnds) * self._vent_factor(v, clat)
        # In Emanuel (2012), Emanuel (2017), there are differing expressions for
        # the intensification rate. It seems Ck and Cd are used interchangeably.
        # Here, in lieu of Ck, we use Cd, and include a variable drag coefficient 
        # to represent faster decay over land.
        Cd = self._get_current_Cd(clon, clat)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            supply = (1 - m) * v
            if self.moisture_noise is not None:
                # Convective organisation and inner-core asymmetries make the supply of inner-core moisture fluctuate.
                # The noise has the moisture equation's own rate scale and vanishes at m = 0 and m = 1, which keeps m in bounds.
                supply += v * np.interp(t, self.noise_t_s, self.moisture_noise) * 4 * m * (1 - m)
            dmdt = 0.5 * Cd / self._h_bl(v, clat) * (supply - venti * m)
        return(dmdt * getattr(namelist, 'moisture_rate_fac', 1.0))

    """ Calculate the steering coefficients. """
    def _calc_steering_coefs(self, v):
        assert len(namelist.steering_coefs) == len(namelist.steering_levels)
        if namelist.coupled_track:
            alpha_fx = (v*1.94384)*np.array(namelist.m_alpha) + np.array(namelist.y_alpha)
            steering_coefs = np.maximum(np.minimum(alpha_fx, namelist.alpha_max), namelist.alpha_min)
            if np.any(np.isnan(steering_coefs)):
                steering_coefs = np.array(namelist.y_alpha)
            steering_coefs = steering_coefs * bam_track.weak_steer_factors(v)
        else:
            steering_coefs = np.array(namelist.steering_coefs)
        return steering_coefs

    """ Time-derivative of the state vector, y.
    y[0] is longitude, y[1] is latitude, y[2] is v, and y[3] is m.
    With ocean_motion_tau_hours, y[4] is the forward speed the ocean feedback answers to."""
    def dydt(self, t, y):
        steering_coefs = self._calc_steering_coefs(y[2])
        v_bam, env_wnds = self._step_bam_track(y[0], y[1], t, steering_coefs, y[2])
        dLondt = v_bam[0] / constants.earth_R * 180. / np.pi / (np.cos(y[1] * np.pi / 180.))
        dLatdt = v_bam[1] / constants.earth_R * 180. / np.pi

        dvdt = self._dvdt(*y[:4], v_bam, env_wnds, t, u_ocean = y[4] if y.size > 4 else None)
        dmdt = self._dmdt(*y[:4], env_wnds, t)
        if self.debug:
            dy = [0, 0, dvdt, dmdt]
        else:
            dy = [dLondt, dLatdt, dvdt, dmdt]
        if y.size > 4:
            dy.append((np.linalg.norm(v_bam) - y[4]) / (namelist.ocean_motion_tau_hours * 3600.))
        return(np.array(dy))

    """
    lon is a 1-D array describing the longitude of the fields: [lon]
    lat is a 1-D array describing the latitude of the fields: [lat]
    chi is a 2-D matrix of chi in space: [lat, lon]
    vpot is a 2-D matrix of potential intensity in space: [lat, lon]
    mld is a 2-D matrix of mixed-layer depth in space: [lat, lon]
    strat is a 2-D matrix of sub-mixed layer thermal stratification in space: [lat, lon]
    """
    def init_fields(self, lon, lat, chi, vpot, mld, strat):
        lon_b, lat_b, chi_b = self.basin.transform_global_field(lon, lat, chi)
        self.f_chi = RectBivariateSpline(lon_b, lat_b, chi_b.T, kx=1, ky=1)
        _, _, vpot_b = self.basin.transform_global_field(lon, lat, vpot)
        self.f_vpot = RectBivariateSpline(lon_b, lat_b, vpot_b.T, kx=1, ky=1)
        _, _, mld_b = self.basin.transform_global_field(lon, lat, mld)
        self.f_mld = RectBivariateSpline(lon_b, lat_b, mld_b.T, kx=1, ky=1)
        _, _, strat_b = self.basin.transform_global_field(lon, lat, strat)
        self.f_strat = RectBivariateSpline(lon_b, lat_b, strat_b.T, kx=1, ky=1)

    """ Generate a track with an initial position of (clon, clat),
        an initial intensity of v, and initial inner core moisture m """
    def gen_track(self, clon, clat, v, m = None):
        # Make sure that tracks are sufficiently randomized.
        bam_track.random_seed()

        # Create the weights for the beta-advection model (across time).
        self.Fs = self.gen_synthetic_f()
        self.Fs_i = interp1d(self.t_s, self.Fs, axis = 1)
        self._gen_noise()

        # If the ventilation index is above some threshold, do not integrate.
        S = self._calc_S(self._env_winds(clon, clat, 0))
        vpot = self._get_current_vpot(clon, clat)
        chi = self._calc_chi(clon, clat)
        if vpot > 0:
            vent_index = S * chi / vpot
            if vent_index >= 1:
                return None

        def tc_dissipates(t, y):
            if not self.basin.in_basin(y[0], y[1], 1):
                # Do not let the track wander outside the basin.
                return 0
            elif np.abs(y[1]) <= 2:
                # Do not let the track wander too equatorward.
                return 0
            else:
                # stopping point when TC reaches 4 m/s
                return np.maximum(0, y[2] - 4)
        tc_dissipates.terminal = True

        # Solve for the intensity.
        if m is None:
            # This means no m has been provided. Initialize with dvdt = 0.
            m_init = self._init_m(np.asarray([clon, clat, v]), 0)
        else:
            m_init = m
        y_init = [clon, clat, v, m_init]
        if getattr(namelist, 'ocean_motion_tau_hours', 0.0) > 0:
            y_init.append(np.linalg.norm(self._step_bam_track(clon, clat, 0., self._calc_steering_coefs(v), v)[0]))
        res = solve_ivp(self.dydt, (0, self.total_time), np.asarray(y_init),
                        t_eval = np.linspace(0, self.total_time, self.total_steps),
                        events = tc_dissipates, max_step = 86400)
        return res

"""
Empirical (IBTrACS)  wobble for synthetic tracks

The wobble is added as a correlated random displacement (an AR(1) process in each direction) 
after the physics (intensity, wind, vmax) has been computed, so it never feeds back into it.
"""
import numpy as np
from scipy.signal import lfilter

from util import constants

def _ar1(n, rho):
    """Stationary, unit-variance AR(1) series of length n with lag-1 correlation rho."""
    burn = int(np.ceil(5.0 / max(1.0 - rho, 1e-3)))
    eps = np.random.standard_normal(n + burn)
    x = lfilter([np.sqrt(1.0 - rho ** 2)], [1.0, -rho], eps)
    return x[burn:]

"""
Displaces a track by an AR(1) wobble with standard deviation sigma_km (per direction) 
and e-folding time tau_h. dt_s is the spacing of the track in seconds.
Returns the displaced (lon, lat).
"""
def apply_wiggle(lon, lat, dt_s, sigma_km, tau_h):
    n = len(lon)
    if sigma_km <= 0 or n < 3:
        return lon, lat
    rho = np.exp(-dt_s / (tau_h * 3600.0)) if tau_h > 0 else 0.0
    km_per_deg = np.pi * constants.earth_R / 1000.0 / 180.0
    dx = sigma_km * _ar1(n, rho)
    dy = sigma_km * _ar1(n, rho)
    lat_w = lat + dy / km_per_deg
    lon_w = lon + dx / (km_per_deg * np.cos(np.deg2rad(lat)))
    return lon_w, lat_w

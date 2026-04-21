
import numpy as np
from util import sphere

# Calculate maximum wind speed from maximum azimuthal wind speed.
def axi_to_max_wind(track_lon, track_lat, dt_track, tc_v, env_wnds):
    utran, vtran = sphere.calc_translational_speed(track_lon, track_lat, dt_track)
    G = np.minimum(1., 0.8 + 0.35 * (1. + np.tanh((track_lat - 35.) / 10.)))
    u_shr = env_wnds[:, 0] - env_wnds[:, 2]
    v_shr = env_wnds[:, 1] - env_wnds[:, 3]
    U_inc = G * utran + 0.1 * u_shr * tc_v / 15.
    V_inc = G * vtran + 0.1 * v_shr * tc_v / 15.

    # Do not allow the wind increment to exceed 50% of the actual intensity.
    mag_inc = np.sqrt(np.power(U_inc, 2) + np.power(V_inc, 2))
    mag_fac = np.minimum(1, (tc_v * 0.50) / mag_inc)
    theta_opt = np.arctan2(-U_inc, V_inc)
    ug = tc_v * -np.sin(theta_opt) + U_inc * mag_fac
    vg = tc_v * np.cos(theta_opt) + V_inc * mag_fac
    tc_vmax = np.sqrt(np.power(ug, 2) + np.power(vg, 2))
    return tc_vmax

# Calculate radius to max wind and radius to gales
def wind_radii(track_lat, tc_vmax):
    rmw = np.exp(4.58 - 0.015 * tc_vmax * 1.94384 + 0.005 * np.abs(track_lat) + np.random.normal(scale=0.39))
    r34 = np.exp(4.08 + 0.008 * tc_vmax * 1.94384 + 0.024 * np.abs(track_lat) + np.random.normal(scale=0.33))
    # Ensure r34 > rmw and r34 < 350km
    r34 = np.where(r34 <= rmw, 1.05*rmw, r34)
    r34 = np.minimum(r34, 350)
    return (rmw, r34)
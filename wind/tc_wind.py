
import numpy as np
import namelist
from track import env_wind
from util import sphere

# Calculate maximum wind speed from maximum azimuthal wind speed. How much motion and shear add to the
# strongest wind is set in the namelist (wind_asym_base, wind_asym_motion, wind_asym_shear, wind_asym_shear_v).
def axi_to_max_wind(track_lon, track_lat, dt_track, tc_v, env_wnds):
    utran, vtran = sphere.calc_translational_speed(track_lon, track_lat, dt_track)
    G = np.minimum(1., namelist.wind_asym_motion + 0.35 * (1. + np.tanh((np.abs(track_lat) - 35.) / 10.)))
    # The part that does not vanish with the motion points along the motion, the floor keeps it finite for a storm at rest
    G = G + namelist.wind_asym_base / np.maximum(np.hypot(utran, vtran), 0.1)
    u250, v250, u850, v850 = env_wind.deep_layer_winds(env_wnds)
    u_shr = u250 - u850
    v_shr = v250 - v850
    shr_fac = namelist.wind_asym_shear * (tc_v / 15. if namelist.wind_asym_shear_v else 1.)
    U_inc = G * utran + shr_fac * u_shr
    V_inc = G * vtran + shr_fac * v_shr

    # Do not allow the wind increment to exceed 50% of the actual intensity.
    mag_inc = np.sqrt(np.power(U_inc, 2) + np.power(V_inc, 2))
    mag_fac = np.minimum(1, (tc_v * 0.50) / mag_inc)
    theta_opt = np.arctan2(-U_inc, V_inc)
    ug = tc_v * -np.sin(theta_opt) + U_inc * mag_fac
    vg = tc_v * np.cos(theta_opt) + V_inc * mag_fac
    tc_vmax = np.sqrt(np.power(ug, 2) + np.power(vg, 2))
    return tc_vmax

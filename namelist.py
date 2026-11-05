import os
import numpy as np

"""
Namelist file that serves as the configuration file for the TC-risk model.
"""

########################## File System Parameters ###########################
src_directory = os.path.dirname(os.path.abspath(__file__))
base_directory = '%s/data/era5' % src_directory
output_directory = '%s/data/era5' % src_directory
exp_name = 'test'
# For now, we support either 'GCM' or 'ERA5'. Different file types and variable
# names can be added by modifying the "input.py" file and adding the appropriate
# variable key words in the structure var_keys.
# file_type is the file type of the input files, and
# can be either 'netcdf' or 'grib'
dataset_type = 'ERA5' #'GCM'
exp_prefix = 'era5' #GFDL-CM4_ssp585_r1i1p1f1'
file_type = 'grib'

# Variable naming based on dataset_type.
# 'sst' is sea-surface temperature (monthly-averaged)
# 'mslp' is mean sea-level pressure (monthly-averaged)
# 'temp' is temperature (monthly-averaged)
# 'sp_hum' is specific humidity (monthly-averaged)
# 'u' is zonal wind (daily)
# 'v' is meridional wind (daily)
var_keys = {'ERA5': {'sst': 'sst', 'mslp': 'sp', 'temp': 't',
                     'sp_hum': 'q', 'u': 'u', 'v': 'v',
                     'lvl': 'isobaricInhPa', 'lon': 'longitude', 'lat': 'latitude'},
            'GCM': {'sst': 'tos', 'mslp': 'psl', 'temp': 'ta',
                    'sp_hum': 'hus', 'u': 'ua', 'v': 'va',
                    'lvl': 'plev', 'lon': 'lon', 'lat': 'lat'}}

# Environmental wind filter (track/env_filter.py) params:
env_wind_sigma_km = 300.0             # width of the Gaussian smoothing, km
env_wind_tc_radius_km = 600.0         # radius removed around each storm, km

# Wind-only storm tracker (util/tc_tracker.py) params:
tc_tracker_grid_deg = 1.0
tc_tracker_params = dict(
    z_track = 2.0e-5,                 # smoothed 850 hPa vorticity a candidate must have, 1/s
    owz = 20e-6,                      # OWZ at 850 hPa a candidate must have nearby, 1/s (0.8 times this at 700 hPa)
    z700_frac = 0.3,                  # smoothed 700 hPa vorticity it must have, as a fraction of that at 850 hPa
    z_strong = 5.0e-5,                # a strong day: smoothed 850 hPa vorticity, 1/s (about 8 m/s tangential wind at 330 km)
    w_strong = 12.0,                  # and largest 850 hPa wind within 3 degrees, m/s
    min_days = 3,                     # days a track must last
    min_strong = 2,                   # strong days over the ocean it must have
    lat0 = 30.0,                      # latitude limit of the first strong day, degrees
    shear0 = 15.0,                    # largest 850-250 hPa shear on the first strong day (over 5 degrees), m/s
    tail_days = 1,                    # days kept after the last strong day
    gap_days = 1,                     # days a track may go unseen
    step_km = 700.0,                  # how far a vortex may move in a day, equatorward of 30 degrees
    step_km_high = 1100.0)            # and poleward of it

########################### Parallelism Parameters ##########################
n_procs = 16              # number of processes to use in dask

############################ TC Risk Parameters #############################
"""
These parameters configure the dates for the TC-risk model.
"""
start_year = 2016                     # year to start downscaling
start_month = 1                       # month of start_year to start downscaling
end_year = 2021                       # year to stop downscaling
end_month = 12                        # month of end_year to stop downscaling

"""
These parameters configure the output.
"""
output_interval_s = 3600              # output interval of tracks, seconds (does not change time integration)
total_track_time_days = 15            # total time to integrate tracks, days
tracks_per_year = 20                  # total number of tracks to simulate per year

"""
These parameters configure thermodynamics and thermodynamic constants.
"""
p_midlevel = 60000
PI_reduc = 0.80
Ck = 1.2e-3
Cd = 1.2e-3
select_thermo = 1   # 1 for pseudoadiabatic, 2 for reversible thermodynamics
select_interp = 2   # 1 for computation, 2 for interpolation
# Empirical correction for slow drift of near-surface relative humidity in ERA5 (only)
pi_rh_drift_years = 3.5

"""
These parameters configure track and intensity constants.
"""
# Defines the steering levels (hPa) of the storm (see paper)
# If 250- and 850-hPa, uses two steering levels.
# If 250-, 500-, and 850-hPa, uses three steering levels.
# The steering_coefficients ('steering_coefs') should have the same 
# length as the number of levels.
steering_levels = [250, 700, 850]
# Weights are fitted on observed TC motion (IBTrACS 1981-2000, checked on 2001-2025) against the monthly environmental winds
steering_coefs = [0.18, 0.42, 0.08]   # constant steering coefficients if not coupled
coupled_track = True                  # track coupled to intensity; overrides alpha
y_alpha = [0.113, 0.434, 0.041]       # value of steering coefficient at 0 knots
m_alpha = [0.00138, -0.00032, 0.00086]  # change of each coefficient per unit storm intensity, 1 / kts
alpha_max = [0.28, 0.44, 0.12]        # maximum value of each steering coefficient (coupled track only)
alpha_min = [0.14, 0.36, 0.05]        # minimum value of each steering coefficient (coupled track only)

# Drift fitted on observed storm motion (IBTrACS 1981-2000, checked on 2001-2025) 
u_drift_nh = ([20, 40, 95, 105, 175, 185, 260, 275], [-0.69, -0.14, -0.14, -1.28, -1.28, -1.29, -1.29, -0.69])
u_drift_sh = ([20, 100, 110, 175, 185, 260], [-1.02, -1.02, 0.2, 0.2, 0.5, 0.5])
u_drift_lat0 = 20.                    # the zonal drift fades to zero between these latitudes
u_drift_lat1 = 40.
v_beta = 1.6                          # poleward drift at low latitudes, m/s
v_beta_slope = 2.5                    # growth of the poleward drift per 30 degrees of latitude beyond drift_lat0, m/s
v_beta_slope_sh = -2.7                # the same in the Southern Hemisphere, where the motion as run asks for less beyond 15 degrees
v_beta_low = 0.8                      # extra poleward drift near the equator, m/s, fading between these latitudes
v_beta_low_lat0 = 6.
v_beta_low_lat1 = 13.
drift_lat0 = 15.
v_drift_nh = ([20, 95, 105, 120, 135, 175, 185, 345], [-0.3, -0.3, 0.0, 0.0, 0.4, 0.4, 0.0, 0.0])
v_drift_sh = ([20, 100, 110, 150, 170, 185, 260, 340], [0.55, 0.55, 0.0, 0.0, 0.7, 1.8, 1.8, 0.55])
steer_fluct_scale = 1.4               # fraction of the synthetic wind fluctuation that steers the storm

# Weak vortices (below TC strength, e.g. developing seeds) are shallower. Factors on the steering weights (one per
# steering level) and drift at these winds, from IBTrACS
weak_vortex_kt = [20.0, 29.5, 34.0]
weak_alpha_fac = [[0.66, 0.84], [0.52, 0.80], [2.8, 2.5]]
weak_u_drift = [-0.6, -0.6]
weak_v_beta = [1.2, 1.2]
weak_v_beta_slope = [1.5, 3.0]

# Storms lean more on the deep flow as they strengthen, but not on its day-to-day swings. Regressing IBTrACS motion on the daily environmental wind anomaly at 250, 700, 850 hPa gives
# about .12, .40, -.05 at every strength, with the total falling from .51 up to 64 kt to .43 at 96-137 kt and .35 above; the spread of motion falls with strength in the same way. So from
# tropical-storm strength on the weights on the swings stay at their value there, times steer_fluct_fac (steer_fluct_kt: where it applies, linear in between, kt). None: the steering weights themselves (the original).
steer_fluct_kt = [64.0, 150.0]
steer_fluct_fac = [1.0, 0.68]

# Empirical 'wiggliness' of the tracks
wiggle_sigma_km = 8.0                 # size of the best-track-like wobble added to tracks, km (0 = off)
wiggle_tau_hours = 6.0                # e-folding time of the wobble, hours

T_days = 30                           # period of the fourier series, days
fluct_spectrum_exponent = 0.7
seed_v_init_ms = 5                    # initial seed v intensity, m/s
seed_v_2d_threshold_ms = 6.5          # seed v threshold after 2 days, m/s
seed_v_threshold_ms = 0.0             # seed v threshold over entire lifetime, m/s (off: observations have no such test)
seed_vmax_threshold_ms = 16.72        # seed vmax threshold over entire lifetime, m/s
                                      # IBTrACS winds are rounded to 5 kt, so 35 kt starts from 32.5 kt
seed_check_days = 2.0                 # time at which a seed must exceed seed_v_2d_threshold_ms, days

# Genesis seeds (util/seeds.py). With seed_density = True, seeds develop in proportion to how often the env
# can host a TC, instead of being spread uniformly with the basin-tuned lat_vort_fac / lat_vort_power
seed_density = True
seed_owz_threshold = 40e-6            # OWZ at 850 hPa on a seed day, 1/s (0.8 times this at 700 hPa) (Tory et al.2013) 
seed_shear_max_ms = 25.0              # largest 850-250 hPa shear on a seed day, m/s
seed_tc_radius_km = 1200.0            # radius left out around each storm, km
seed_smooth_deg = 1.5                 # Gaussian smoothing of the monthly seed-day frequency, degrees
seed_density_ref = 0.6                # seed-day frequency at which every draw would develop (only sets run time)

# Atmospheric boundary layer depth (FAST), in meters. Sets how fast storms intensify and decay (rate ~ 1/depth)
atm_bl_depth = {'NA': 2200.0, 'EP': 2200.0, 'WP': 2200.0, 'AU': 2200.0,
                'SI': 2200.0, 'SP': 2200.0, 'NI': 2200.0}
bl_v_ref_ms = 18.0                    # intensity below which ventilation (and the depth above) ignores v, m/s
bl_rmw_km = 50.0                      # radius of maximum wind used for the inertial stability, km
log_chi_fac = 0.5                     # addition to chi in log space
chi_fac = 1.3                         # addition to chi
vent_inertial_exponent = 0.75
vent_radius_km = 150.0
moisture_rate_fac = 1.57
spindown_fac = 1.5

# Upper-ocean maps of the ocean feedback, from World Ocean Atlas 2023 (scripts/build_ocean_maps.py), 
# instead of the original Levitus (1982)
ocean_mld_file = '%s/intensity/data/mld_climatology_woa23.nc' % src_directory
ocean_strat_file = '%s/intensity/data/strat_climatology_woa23.nc' % src_directory
# Latitude dependence of the ocean feedback, z ~ f^ocean_f_exponent (Schade and Emanuel 1999).
ocean_f_exponent = 0.59
shelf_radius_km = 100.0
ocean_motion_tau_hours = 12.0

# Wind asymmetry parameters: How much motion and shear add to the strongest wind
# Values are the amplitudes measured in TC wind fields by Chang, Amin and Emanuel (2020): 1.17 + 0.40 x speed and about 0.1 x shear
wind_asym_base = 1.17
wind_asym_motion = 0.40
wind_asym_shear = 0.1
wind_asym_shear_v = False

# Storm crowding (util/crowding.py): the share of new storms suppressed at these distances from an active storm
# Empirical fit from IBTrACS obs (1981-2025) 
crowd_km = [250.0, 750.0, 1250.0, 1600.0]
crowd_suppress = [0.98, 0.85, 0.34, 0.0]
crowd_storms_per_year = 90.8
crowd_constant = None

lat_vort_fac = 2                      # sets where vorticity threshold decays toward equator
lat_vort_power = {'NA': 6, 'EP': 6,   # power decay towards the equator
                  'WP': 3.5, 'AU': 6,
                  'SI': 3, 'SP': 7, 'NI': 2.5}
# Initial m based on large-scale relative humidity
f_mInit = lambda rh: 0.20 / (1 + np.exp(-(rh - 0.55) * 10)) + 0.125

"""
Basins for which the model is enabled.
The basin bounds dictionary maps a basin identifier to the basin boundaries.
The bounds are ordered as (LL - Lower Left, UR - Upper Right):
[LL Longitude, LL Latitude, UR Longitude, UR Latitude].
Note the basins bounds are extended slightly to allow tracks to
extend slightly beyond the bounds, as this may be the case when a TC starts
in one basin and goes to another.
Identifiers: EP - Eastern Pacific
             NA - North Atlantic
             NI - North Indian
             SI - South Indian
             SP - South Pacific
             WP - Western Pacific
             GL - Global (no basin)
"""
basin_bounds = {'EP': ['180E', '0N', '290E', '60N'],
                'NA': ['260E', '0N', '360E', '60N'],
                'NI': ['30E', '0N', '100E', '50N'],
                'SI': ['20E', '45S', '100E', '0S'],
                'AU': ['100E', '45S', '180E', '0S'],
                'SP': ['180E', '45S', '250E', '0S'],
                'WP': ['100E', '0N', '180E', '60N'],
                'GL': ['0E', '90S', '360E', '90N']}


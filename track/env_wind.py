import dask
import datetime
import numpy as np
import os
import xarray as xr

import namelist
from dask.distributed import LocalCluster, Client
from util import input
import metpy.calc as mpcalc
from metpy.units import units

"""
Returns the name of the file containing environmental wind statistics.
"""
def get_env_wnd_fn():
    fn_out = '%s/env_wnd_%s_%d%02d_%d%02d.nc' % (namelist.output_directory, namelist.exp_prefix,
                                                 namelist.start_year, namelist.start_month,
                                                 namelist.end_year, namelist.end_month)
    return(fn_out)

"""
Deterministic ordered list of all levels for which we store mean winds
(genesis levels + steering levels, preserving genesis_levels order).
"""
def _all_wind_levels():
    lvls = list(namelist.genesis_levels)
    for x in namelist.steering_levels:
        if x not in lvls:
            lvls.append(x)
    return lvls

"""
Generates variable names in the monthly mean wind vector.
"""
def wind_mean_vector_names():
    p_lvls = _all_wind_levels()
    var_names = sum([['ua%s' % x, 'va%s' % x] for x in p_lvls], [])
    var_Mean = [x + '_Mean' for x in var_names]
    return var_Mean

"""
Names of the mean wind components that pair with the covariance matrix
(steering levels only, covariance is computed only for these).
"""
def steering_wind_mean_names():
    p_lvls = namelist.steering_levels
    var_names = sum([['ua%s' % x, 'va%s' % x] for x in p_lvls], [])
    return [x + '_Mean' for x in var_names]

"""
Generate list of variable names for monthly mean gradients.
"""
def wind_gradient_names():
    var_Gradient = ['dudyDLM_Mean', 'dvdxDLM_Mean', 'dzdyDLM_Mean']
    return var_Gradient

"""
Generates variable names in the monthly wind covariance matrix.
"""
def wind_cov_matrix_names():
    p_lvls = namelist.steering_levels
    var_names = sum([['ua%s' % x, 'va%s' % x] for x in p_lvls], [])
    var_Mean = [x + '_Mean' for x in var_names]
    var_Var = [['' for i in range(len(var_names))] for j in range(len(var_names))]
    for i in range(len(var_names)):
        for j in range(0, i+1):
            if i == j:
                var_Var[i][j] = var_names[i] + '_Var'
            else:
                var_Var[i][j] = var_names[i] + '_' + var_names[j] + '_cov'
    return var_Var

"""
Extracts the deep-layer from the wind vector 'env_wnds',
which is assumed to be a 2-D array of dimensions (time, nWLvl).
Used to generate deep-layer shear.
"""
def deep_layer_winds(env_wnds):
    var_names = steering_wind_mean_names()
    u250 = env_wnds[:, var_names.index('ua250_Mean')]
    v250 = env_wnds[:, var_names.index('va250_Mean')]
    u850 = env_wnds[:, var_names.index('ua850_Mean')]
    v850 = env_wnds[:, var_names.index('va850_Mean')]
    return (u250, v250, u850, v850)

"""
Read the mean and covariance of the upper/lower level zonal and meridional winds.
"""
def read_env_wnd_fn(fn_wnd_stat, dt_s = None, dt_e = None):
    var_Mean = steering_wind_mean_names()
    var_Var = wind_cov_matrix_names()
    var_Grad = wind_gradient_names()

    if dt_s is None:
        ds = xr.open_dataset(fn_wnd_stat).sortby("lat", ascending=True).sel(lat=slice(-65,65))
    else:
        ds = xr.open_dataset(fn_wnd_stat).sortby("lat", ascending=True).sel(lat=slice(-65,65)).sel(time = slice(dt_s, dt_e))
    wnd_Mean = [ds[x] for x in var_Mean]
    wnd_Grad = [ds[x] for x in var_Grad]
    wnd_Cov = [['' for i in range(len(var_Mean))] for j in range(len(var_Mean))]
    for i in range(len(var_Mean)):
        for j in range(len(var_Mean)):
            if j > i:
                wnd_Cov[i][j] = ds[var_Var[j][i]]
            else:
                wnd_Cov[i][j] = ds[var_Var[i][j]]

    return (wnd_Mean, wnd_Cov)

"""
Generate the wind mean and covariance matrices used to advect
tropical cyclones.
"""
def gen_wind_mean_cov():
    fn_out = get_env_wnd_fn()
    if os.path.exists(fn_out):
        return

    # Since the operations are massively parallelized, we want individual
    # control over the files being opened.
    dt_start, dt_end = input.get_bounding_times()
    fns_ua = input._find_in_timerange(input._glob_prefix(input.get_u_key()), dt_start, dt_end)
    fns_va = input._find_in_timerange(input._glob_prefix(input.get_v_key()), dt_start, dt_end) 
    n_files = min(len(fns_ua), len(fns_va))
    if namelist.use_dask:
        cl_args = {'n_workers': namelist.n_procs,
                    'processes': True,
                    'threads_per_worker': 1}
        lazy_results = []
        with LocalCluster(**cl_args) as cluster, Client(cluster) as client:
            for i in range(min(len(fns_ua), len(fns_va))):
                lazy_result = dask.delayed(wnd_stat_wrapper)((fns_ua[i], fns_va[i]))
                lazy_results.append(lazy_result)
            out = dask.compute(*lazy_results)
    else:
        out = [wnd_stat_wrapper((fns_ua[i], fns_va[i])) for i in range(n_files)]
    out_fns = [x for x in out if x is not None]

    # Combine all intermediate files into one dataset, and delete
    ds = input._open_fns(out_fns)
    da = ds['wnd_stats'].load()

    var_Mean = wind_mean_vector_names()
    var_Var = sum([[x for x in y if len(x) > 0] for y in wind_cov_matrix_names()], [])
    var_names = var_Mean + var_Var + ['dudyDLM_Mean', 'dvdxDLM_Mean', 'dzdyDLM_Mean']
    var_dict = dict()
    for i in range(len(var_names)):
        var_dict[var_names[i]] = da[:, i, :, :].rename(var_names[i])
    ds.close()
    del ds, da

    ds_e = xr.Dataset(data_vars = var_dict)
    try:
        ds_e.to_netcdf(fn_out, mode='w', engine='h5netcdf')
    except Exception:
        ds_e.to_netcdf(fn_out, mode='w', engine='netcdf4')
    print('Saved %s' % fn_out)

    # Remove intermediate files
    for fn in out_fns:
        os.remove(fn)

def wnd_stat_wrapper(args):
    fn_u, fn_v = args

    dt_start, dt_end = input.get_bounding_times()
    ds_ua = input._load_var_daily(fn_u).sortby("latitude", ascending=True).sel(latitude=slice(-65,65))
    ds_va = input._load_var_daily(fn_v).sortby("latitude", ascending=True).sel(latitude=slice(-65,65))
    ua = ds_ua[input.get_u_key()]
    va = ds_va[input.get_v_key()]
    
    # Find all of the months to average over.
    dts = input.convert_to_datetime(ds_ua, ds_ua['time'].values)
    dt_start = max([dt_start, dts[0]])
    dt_start = datetime.datetime(dt_start.year, dt_start.month, 15)
    t_months = [dt_start]
    while t_months[-1] <= min([dt_end, dts[-1]]):
        cYear = t_months[-1].year
        cMonth = t_months[-1].month
        if cMonth == 12:
            t_months.append(datetime.datetime(cYear + 1, 1, 15))
        else:
            t_months.append(datetime.datetime(cYear, cMonth + 1, 15))

    t_months = t_months[0:-1]
    nMonths = len(t_months)

    # No computations found for this file.
    if nMonths == 0:
        return

    # Compute mean and covariances for upper and lower level horizontal winds.
    out = [0]*nMonths
    for i in range(nMonths):
        out[i] = calc_wnd_stat(ua, va, t_months[i])

    # Save the results using an intermediate file.
    da_wnd = xr.DataArray(data = xr.concat(out, dim = "time").data,
                          dims = ['time', 'stat', 'lat', 'lon'],
                          coords = dict(lon = ("lon", ds_ua[input.get_lon_key()].data),
                                        lat = ("lat", ds_ua[input.get_lat_key()].data),
                                        time = ("time", input.convert_from_datetime(ds_ua, t_months).astype('datetime64[ns]'))))
    ds_wnd = da_wnd.to_dataset(name='wnd_stats')
    fn_ds_wnd = '%s/env_wnd_%s_p%d%02d_%d%02d.nc' % (namelist.output_directory, namelist.exp_prefix,
                                                     t_months[0].year, t_months[0].month,
                                                     t_months[-1].year, t_months[-1].month)
    try:
        ds_wnd.to_netcdf(fn_ds_wnd, engine='h5netcdf')
    except Exception:
        ds_wnd.to_netcdf(fn_ds_wnd, engine='netcdf4')
    return fn_ds_wnd

"""
Computes mean and covariance of environmental winds across a month.
"""
def calc_wnd_stat(ua, va, dt):
    cYear = dt.year
    cMonth = dt.month

    if cMonth == 12:
        tEnd = datetime.datetime(cYear + 1, 1, 1)
    else:
        tEnd = datetime.datetime(cYear, cMonth + 1, 1)
    month_mask = ((input.convert_to_datetime(ua, ua['time'].values) >= datetime.datetime(cYear, cMonth, 1)) &
                  (input.convert_to_datetime(ua, ua['time'].values) < tEnd))

    lvl = ua[input.get_lvl_key()]
    # if lvl.units in ['millibars', 'hPa']:
    #     p_upper = 250; p_lower = 850;
    # else:
    #     p_upper = 25000; p_lower = 85000;
    p_upper, p_lower = namelist.steering_levels
    all_levels = _all_wind_levels()


    # If time step is less than one day, group by day.
    dt_step = (np.timedelta64(1, 'D') - (ua['time'][1] - ua['time'][0]).data) / np.timedelta64(1, 's')
    if dt_step < 0:
        ua_month = ua.sel(time = month_mask).groupby("time.day").mean(dim = 'time')
        va_month = va.sel(time = month_mask).groupby("time.day").mean(dim = 'time')
        t_unit = 'day'
    else:
        ua_month = ua.sel(time = month_mask)
        va_month = va.sel(time = month_mask)
        t_unit = 'time'

    # Compute the daily averages
    # ua250_month = ua_month.sel({input.get_lvl_key(): p_upper})
    # va250_month = va_month.sel({input.get_lvl_key(): p_upper})
    # ua850_month = ua_month.sel({input.get_lvl_key(): p_lower})
    # va850_month = va_month.sel({input.get_lvl_key(): p_lower})

    # month_wnds = [ua250_month, va250_month,
    #               ua850_month, va850_month]

    # Means for ALL levels (order matches wind_mean_vector_names).
    all_wnds = []
    for lev in all_levels:
        all_wnds.append(ua_month.sel({input.get_lvl_key(): lev}, method='nearest'))
        all_wnds.append(va_month.sel({input.get_lvl_key(): lev}, method='nearest'))
    month_mean_wnds = [w.mean(dim=t_unit) for w in all_wnds]

    # Covariances for STEERING levels only (order matches wind_cov_matrix_names).
    steering_wnds = []
    for lev in namelist.steering_levels:
        steering_wnds.append(ua_month.sel({input.get_lvl_key(): lev}, method='nearest'))
        steering_wnds.append(va_month.sel({input.get_lvl_key(): lev}, method='nearest'))


    n_s = len(steering_wnds)
    cov_entries = []
    for i in range(n_s):
        for j in range(0, i+1):
            if i == j:
                cov_entries.append(steering_wnds[i].var(dim=t_unit))
            else:
                cov_entries.append(xr.cov(steering_wnds[i], steering_wnds[j], dim=t_unit))

    # Deep-layer mean relative vorticity (3 entries).
    ua_dlm_u = ua_month.sel({input.get_lvl_key(): p_upper})
    va_dlm_u = va_month.sel({input.get_lvl_key(): p_upper})
    ua_dlm_l = ua_month.sel({input.get_lvl_key(): p_lower})
    va_dlm_l = va_month.sel({input.get_lvl_key(): p_lower})
    uadlm = 0.8 * ua_dlm_l + 0.2 * ua_dlm_u
    vadlm = 0.8 * va_dlm_l + 0.2 * va_dlm_u
    month_mean_vort = list(compute_mean_vorticity(uadlm, vadlm))

    stats = month_mean_wnds + cov_entries + month_mean_vort

    wnd_stats = np.zeros((len(stats),) + month_mean_wnds[0].shape)
    for i in range(len(stats)):
        wnd_stats[i, :, :] = stats[i]

    wnd_stats = xr.DataArray(
            data = wnd_stats,
            dims = ["stat", "lat", "lon"],
            coords = dict(
                lon=(ua[input.get_lon_key()].values),
                lat=(ua[input.get_lat_key()].values)))

    return wnd_stats

def compute_mean_vorticity(ua, va,):
    """
    Compute mean vorticity of a wind field
    :param ua: `xr.DataArray` of u-component of wind
    :param va: `xr.DataArray` of v-component of wind
    
    :return: meridional gradient of zonal wind, zonal gradient of 
    meridional wind and meridional gradient of vorticity. 
    
    These values are scaled by 10^5, 10^5 and 10^11 respectively, 
    for use in the steering flow calculations. These values are not
    used in the calculation of the genesis parameter.
    
    Wind data are smoothed with two passes of a 9-point smoother before
    calculating vorticity & gradients.
    """
    ua = ua.metpy.dequantify() * units('m/s')
    va = va.metpy.dequantify() * units('m/s')

    uasm = mpcalc.smooth_n_point(ua, 9, 2)
    vasm = mpcalc.smooth_n_point(va, 9, 2)
    
    vrt = mpcalc.vorticity(uasm, vasm)
    dzdy, dzdx = mpcalc.gradient(
        vrt.mean(dim="time"),
        axes=[input.get_lat_key(),
              input.get_lon_key()]
        )
    dudy, dudx = mpcalc.gradient(
        uasm.mean(dim="time"),
        axes=[input.get_lat_key(),
              input.get_lon_key()]
        )
    dvdy, dvdx = mpcalc.gradient(
        vasm.mean(dim="time"),
        axes=[input.get_lat_key(),
              input.get_lon_key()]
        )
    dudy = dudy.assign_coords({'level': 850})
    dvdx = dvdx.assign_coords({'level': 850})
    dzdy = dzdy.assign_coords({'level': 850})

    return dudy*10e5, dvdx*10e5, dzdy*10e11

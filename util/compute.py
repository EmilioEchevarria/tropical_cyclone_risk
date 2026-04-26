#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jzlin@mit.edu
"""
import os
import dask
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import time

import namelist
from dask.distributed import LocalCluster, Client, as_completed
from intensity import coupled_fast, ocean
from thermo import calc_thermo
from track import env_wind
from wind import tc_wind
from util import basins, input, mat
from genesis import calc_genesis
from scipy.ndimage import maximum_filter

"""
Driver function to compute zonal and meridional wind monthly mean and
covariances, potential intensity, GPI, and saturation deficit.
"""
def compute_downscaling_inputs():
    if namelist.file_type == 'grib':
        print('Preprocessing .grib files...')
        input.preprocess_grib()
    
    print('Computing monthly mean and variance of environmental wind...')
    s = time.time()
    env_wind.gen_wind_mean_cov()
    e = time.time()
    print('Time Elapsed: %f s' % (e - s))

    print('Computing thermodynamic variables...')
    s = time.time()
    calc_thermo.gen_thermo()
    e = time.time()
    print('Time Elapsed: %f s' % (e - s))

    print("Computing genesis parameter")
    s = time.time()
    calc_genesis.gen_genesis()
    e = time.time()
    print('Time Elapsed: %f s' % (e - s))

"""
Returns the name of the file containing downscaled tropical cyclone tracks.
"""
def get_fn_tracks(b):
    fn_args = (namelist.output_directory, namelist.exp_name,
               b.basin_id, namelist.exp_prefix,
               namelist.start_year, namelist.start_month,
               namelist.end_year, namelist.end_month)
    fn_trk = '%s/%s/tracks_%s_%s_%d%02d_%d%02d.nc' % fn_args
    return(fn_trk)

"""
Adds a number to the end of fn_trk if the file exists.
Used when running multiple simulations under the same configuration.
"""
def fn_tracks_duplicates(fn_trk):
    f_int = 0
    fn_trk_out = fn_trk
    while os.path.exists(fn_trk_out):
        fn_trk_out = fn_trk.rstrip('.nc') + '_e%d.nc' % f_int
        f_int += 1
    return fn_trk_out

"""
Filename for a single-year checkpoint file (used when namelist.save_yearly=True)
"""
def get_fn_tracks_yearly(b, year):
    fn_args = (namelist.output_directory, namelist.exp_name,
               b.basin_id, namelist.exp_prefix, year)
    return '%s/%s/tracks_%s_%s_%d.nc' % fn_args

"""
Build an xr.Dataset from one or more run_tracks() return tuples. out_list is a list of the per-year tuples 
years is a parallel list of the year each tuple corresponds to
"""
def _build_tracks_dataset(out_list, years, basin_ids):
    tc_lon = np.concatenate([x[0] for x in out_list], axis=0)
    tc_lat = np.concatenate([x[1] for x in out_list], axis=0)
    tc_v = np.concatenate([x[2] for x in out_list], axis=0)
    tc_m = np.concatenate([x[3] for x in out_list], axis=0)
    tc_vmax = np.concatenate([x[4] for x in out_list], axis=0)
    tc_rmax = np.concatenate([x[5] for x in out_list], axis=0)
    tc_r34 = np.concatenate([x[6] for x in out_list], axis=0)
    tc_env_wnds = np.concatenate([x[7] for x in out_list], axis=0)
    tc_months = np.concatenate([x[8] for x in out_list], axis=0)
    tc_basins = np.concatenate([x[9] for x in out_list], axis=0)
    tc_years = np.concatenate([[years[i]] * out_list[i][0].shape[0]
                               for i in range(len(out_list))], axis=0)
    n_seeds = np.array([x[10] for x in out_list])

    total_time_s = namelist.total_track_time_days * 24 * 60 * 60
    n_steps_output = int(total_time_s / namelist.output_interval_s) + 1
    ts_output = np.linspace(0, total_time_s, n_steps_output)

    ds = xr.Dataset(
        data_vars=dict(lon_trks=(["n_trk", "time"], tc_lon),
                       lat_trks=(["n_trk", "time"], tc_lat),
                       u250_trks=(["n_trk", "time"], tc_env_wnds[:,:,0]),
                       v250_trks=(["n_trk", "time"], tc_env_wnds[:,:,1]),
                       u850_trks=(["n_trk", "time"], tc_env_wnds[:,:,2]),
                       v850_trks=(["n_trk", "time"], tc_env_wnds[:,:,3]),
                       v_trks=(["n_trk", "time"], tc_v),
                       m_trks=(["n_trk", "time"], tc_m),
                       vmax_trks=(["n_trk", "time"], tc_vmax),
                       rmax_trks=(["n_trk", "time"], tc_rmax),
                       r34_trks=(["n_trk", "time"], tc_r34),
                       tc_month=(["n_trk"], tc_months),
                       tc_basins=(["n_trk"], tc_basins),
                       tc_years=(["n_trk"], tc_years),
                       seeds_per_month=(["year", "basin", "month"], n_seeds)),
        coords=dict(n_trk=range(tc_lon.shape[0]), time=ts_output,
                    year=np.asarray(years), basin=basin_ids,
                    month=list(range(1, 13))))
    return ds

def _log(msg):
    print('[%s] [run_tracks pid=%d] %s' % (
        datetime.datetime.now().strftime('%H:%M:%S'), os.getpid(), msg), flush=True)

"""
Generates "n_tracks" number of tropical cyclone tracks, in basin
described by "b" (can be global), in the year.
"""
def run_tracks(year, n_tracks, b):
    t0 = time.time()
    _log('starting year=%d basin=%s' % (year, b.basin_id))

    # Load thermodynamic and ocean variables.
    fn_th = calc_thermo.get_fn_thermo()
    ds = xr.open_dataset(fn_th).sortby("lat", ascending=True).sel(lat=slice(-65,65))
    dt_year_start = datetime.datetime(year-1, 12, 31)
    dt_year_end = datetime.datetime(year, 12, 31)
    dt_bounds = input.convert_from_datetime(ds, [dt_year_start, dt_year_end])
    ds = ds.sel(time=slice(dt_bounds[0], dt_bounds[1])).load()
    lon = ds['lon'].data
    lat = ds['lat'].data
    mld = ocean.mld_climatology(year, basins.TC_Basin('GL'))
    strat = ocean.strat_climatology(year, basins.TC_Basin('GL'))    # Make sure latitude is increasing.
    vpot = ds['vmax'] * namelist.PI_reduc * np.sqrt(namelist.Ck / namelist.Cd)
    rh_mid = ds['rh_mid']
    chi = ds['chi']

    if (lat[0] - lat[1]) > 0:
        vpot = vpot.reindex({'lat': lat[::-1]})
        rh_mid = rh_mid.reindex({'lat': lat[::-1]})
        chi = chi.reindex({'lat': lat[::-1]})
        lat = lat[::-1]

    # Load genesis parameter:
    fn_tcgp = calc_genesis.get_fn_tcgp()
    gds = xr.open_dataset(fn_tcgp)
    gds = gds.sel(time=slice(dt_bounds[0], dt_bounds[1])).load()
    glon = gds['lon'].data
    glat = gds['lat'].data
    gds['tcgp'] = gds['tcgp']/np.nanpercentile(gds['tcgp'].values.flatten(), 99.9)
    tcgp = gds['tcgp']

    if (glat[0] - glat[1]) > 0:
        tcgp = tcgp.reindex({'lat': glat[::-1]})
        glat = glat[::-1]

    # Load the basin bounds and genesis points.
    basin_ids = np.array(sorted([k for k in namelist.basin_bounds if k != 'GL']))
    f_basins = {}
    for basin_id in basin_ids:
        ds_b = xr.open_dataset('land/%s.nc' % basin_id)
        basin_mask = ds_b['basin']
        f_basins[basin_id] = mat.interp2_fx(basin_mask['lon'], basin_mask['lat'], basin_mask)

    # In case basin is "GL", we load again.
    ds_b = xr.open_dataset('land/%s.nc' % b.basin_id)
    basin_mask = ds_b['basin']
    f_b = mat.interp2_fx(basin_mask['lon'], basin_mask['lat'], basin_mask)
    b_bounds = b.get_bounds()

    # To randomly seed in both space and time, load data for each month in the year.
    cpl_fast = [0] * 12
    m_init_fx = [0] * 12
    tcgp_month = [0] * 12
    tcgp_init_fx = [0] * 12
    tcgp_dilated_fx = [0] * 12

    n_seeds = np.zeros((len(basin_ids), 12))
    T_s = namelist.total_track_time_days * 24 * 60 * 60     # total time to run tracks
    fn_wnd_stat = env_wind.get_env_wnd_fn()
    ds_wnd = xr.open_dataset(fn_wnd_stat).sortby("lat", ascending=True).sel(lat=slice(-65,65))

    for i in range(12):
        dt_month = datetime.datetime(year, i + 1, 15)
        ds_dt_month = input.convert_from_datetime(ds_wnd, [dt_month])[0].astype('datetime64[ns]')
        vpot_month = np.nan_to_num(vpot.sel(time = ds_dt_month, method='nearest').data, 0)
        rh_mid_month = rh_mid.sel(time = ds_dt_month, method='nearest').data
        chi_month = chi.sel(time = ds_dt_month, method='nearest').data
        chi_month[np.isnan(chi_month)] = 5
        m_init_fx[i] = mat.interp2_fx(lon, lat, rh_mid_month)
        chi_month = np.maximum(np.minimum(np.exp(np.log(chi_month + 1e-3) + namelist.log_chi_fac) + namelist.chi_fac, 5), 1e-5)

        mld_month = mat.interp_2d_grid(mld['lon'], mld['lat'], np.nan_to_num(mld[:, :, i]), lon, lat)
        strat_month = mat.interp_2d_grid(strat['lon'], strat['lat'], np.nan_to_num(strat[:, :, i]), lon, lat)
        cpl_fast[i] = coupled_fast.Coupled_FAST(fn_wnd_stat, b, ds_dt_month,
                                                namelist.output_interval_s, T_s)
        cpl_fast[i].init_fields(lon, lat, chi_month, vpot_month, mld_month, strat_month)

        gds_dt_month = input.convert_from_datetime(gds, [dt_month])[0]
        tcgp_month[i] = np.nan_to_num(tcgp.sel(time=gds_dt_month, method='nearest').data, 0)
        tcgp_init_fx[i] = mat.interp2_fx(tcgp['lon'], tcgp['lat'], tcgp_month[i])

        # Dilated-max tcgp field for cheap seed pre-filter. A TC seed can only reach TS-strength genesis 
        # within a plausible translation radius. we reject seeds where the neighborhood-max tcgp is zero 
        # so we skip the expensive ODE integration for seeds that could not pass the strict
        # post-integration tcgp check anyway
        tcgp_dilation_radius_deg = 5.0
        dlat = float(np.abs(np.mean(np.diff(tcgp['lat'].data))))
        dlon = float(np.abs(np.mean(np.diff(tcgp['lon'].data))))
        ksize_lat = int(np.ceil(tcgp_dilation_radius_deg / dlat)) * 2 + 1
        ksize_lon = int(np.ceil(tcgp_dilation_radius_deg / dlon)) * 2 + 1
        tcgp_dilated = maximum_filter(tcgp_month[i], size=(ksize_lat, ksize_lon), mode='nearest')
        tcgp_dilated_fx[i] = mat.interp2_fx(tcgp['lon'], tcgp['lat'], tcgp_dilated)

    max_tcgp_month = np.nanmax(tcgp_month)

    _log('year=%d setup done in %.1fs. Starting track generation (%d tracks)' %
         (year, time.time() - t0, n_tracks))
    t_tracks = time.time()

    # Output vectors.
    nt = 0
    n_steps = cpl_fast[0].total_steps
    tc_lon = np.full((n_tracks, n_steps), np.nan)
    tc_lat = np.full((n_tracks, n_steps), np.nan)
    tc_v = np.full((n_tracks, n_steps), np.nan)
    tc_m = np.full((n_tracks, n_steps), np.nan)
    tc_vmax = np.full((n_tracks, n_steps), np.nan)
    tc_rmax = np.full((n_tracks, n_steps), np.nan)
    tc_r34 = np.full((n_tracks, n_steps), np.nan)
    tc_env_wnds = np.full((n_tracks, n_steps, cpl_fast[0].nWLvl), np.nan)
    tc_month = np.full(n_tracks, np.nan)
    tc_basin = np.full(n_tracks, "", dtype = 'U2')

    seeds_df_list = []
    tcs_df_list = []

    next_pct = 10
    while nt < n_tracks:
        # "Progress bar":
        pct = 100 * nt / n_tracks
        if pct >= next_pct:
            print(f"{next_pct}% done")
            next_pct += 10
        
        seed_passed = False
        while not seed_passed:
            # Random genesis location for the seed (weighted by area).
            # Ensure that it is located within the basin and over ocean.
            # Genesis is [3, 45] latitude for each basin.
            lat_min = 3 if np.sign(b_bounds[1]) >= 0 else -45
            lat_max = 45 if np.sign(b_bounds[3]) >= 0 else -3
            y_min = np.sin(np.pi / 180 * lat_min)
            y_max = np.sin(np.pi / 180 * lat_max)
            gen_lon = np.random.uniform(b_bounds[0], b_bounds[2], 1)[0]
            gen_lat = np.arcsin(np.random.uniform(y_min, y_max, 1)[0]) * 180 / np.pi
            while f_b.ev(gen_lon, gen_lat) < 1e-2:
                gen_lon = np.random.uniform(b_bounds[0], b_bounds[2], 1)[0]
                gen_lat = np.arcsin(np.random.uniform(y_min, y_max, 1)[0]) * 180 / np.pi

            # Randomly seed the month.
            month_seed = np.random.randint(1, 13)
            fast = cpl_fast[month_seed - 1]

            # Find basin of genesis location and switch H_bl.
            basin_val = np.zeros(len(basin_ids))
            for (b_idx, basin_id) in enumerate(basin_ids):
                basin_val[b_idx] = f_basins[basin_id].ev(gen_lon, gen_lat)
            basin_idx = np.argmax(basin_val)

            # Discard seeds with increasing probability equatorwards.
            # If PI is less than 35 m/s, do not integrate, but treat as a seed.
            pi_gen = float(fast.f_vpot.ev(gen_lon, gen_lat))
            lat_vort_power = namelist.lat_vort_power[basin_ids[basin_idx]]
            prob_lowlat = np.power(np.minimum(np.maximum((np.abs(gen_lat) - namelist.lat_vort_fac) / 12.0, 0), 1), lat_vort_power)
            rand_lowlat = np.random.uniform(0, 1, 1)[0]
            if (np.nanmax(basin_val) > 1e-3) and (rand_lowlat < prob_lowlat):
                n_seeds[basin_idx, month_seed-1] += 1
                seed_df = pd.DataFrame([[gen_lat,gen_lon,month_seed,year]],columns=['lat','lon','month','year'])
                seeds_df_list.append(seed_df)
                if (pi_gen > 35):
                    tcgp_nbhd = float(tcgp_dilated_fx[month_seed - 1].ev(gen_lon, gen_lat))
                    if tcgp_nbhd > 0:
                        seed_passed = True

        # Set the initial value of m to a function of relative humidity.
        v_init = namelist.seed_v_init_ms + np.random.randn(1)[0]
        rh_init = float(m_init_fx[month_seed-1].ev(gen_lon, gen_lat))
        m_init = np.maximum(0, namelist.f_mInit(rh_init))
        fast.h_bl = namelist.atm_bl_depth[basin_ids[basin_idx]]
        res = fast.gen_track(gen_lon, gen_lat, v_init, m_init)

        is_tc = False
        if res != None:
            track_lon = res.y[0]
            track_lat = res.y[1]
            v_track = res.y[2]
            m_track = res.y[3]

            # If the TC has not reached the threshold m/s after 2 days, throw it away.
            # The TC must also reach the genesis threshold during it's entire lifetime.
            v_thresh = namelist.seed_v_threshold_ms
            v_thresh_2d = np.interp(2*24*60*60, res.t, v_track.flatten())

            # TCGP check for genesis location
            genesis_v_thresh = 19.88  # 17/0.88 m/s — tropical storm threshold 
            genesis_idx = np.where(v_track.flatten() >= genesis_v_thresh)[0]
            tcgp_passed = False
            if len(genesis_idx) > 2:
                gi = genesis_idx[0]
                genesis_lon = track_lon[gi]
                genesis_lat = track_lat[gi]
                tcgp_genesis = tcgp_init_fx[month_seed - 1].ev(genesis_lon, genesis_lat)
                rand_tcgp = np.random.uniform(0, max_tcgp_month, 1)[0]
                tcgp_passed = (rand_tcgp < tcgp_genesis)
            
            #is_tc = np.logical_and(np.any(v_track >= v_thresh), v_thresh_2d >= namelist.seed_v_2d_threshold_ms)
            is_tc = np.logical_and.reduce([
                np.any(v_track >= v_thresh),
                v_thresh_2d >= namelist.seed_v_2d_threshold_ms,
                np.min(np.abs(track_lat)) > 2.0,
                tcgp_passed])

        if is_tc:
            n_time = len(track_lon)
            tc_lon[nt, 0:n_time] = track_lon
            tc_lat[nt, 0:n_time] = track_lat
            tc_v[nt, 0:n_time] = v_track
            tc_m[nt, 0:n_time] = m_track

            # Redudant calculation, but since environmental winds are not part
            # of the time-integrated state (a parameter), we recompute it.
            # TODO: Remove this redudancy by pre-caclulating the env. wind.
            for i in range(len(track_lon)):
                tc_env_wnds[nt, i, :] = fast._env_winds(track_lon[i], track_lat[i], fast.t_s[i])     
            vmax = tc_wind.axi_to_max_wind(track_lon, track_lat, fast.dt_track,
                                           v_track, tc_env_wnds[nt, 0:n_time, :])
            rmax, r34 = tc_wind.wind_radii(track_lat, vmax)
            if np.nanmax(vmax) >= namelist.seed_vmax_threshold_ms:
                tc_vmax[nt, 0:n_time] = vmax
                tc_rmax[nt, 0:n_time] = rmax
                tc_r34[nt, 0:n_time] = r34
                tc_month[nt] = month_seed
                tc_basin[nt] = basin_ids[basin_idx]
                nt += 1
                _log('year=%d - accepted track %d/%d (elapsed %.1fs)' %
                     (year, nt, n_tracks, time.time() - t_tracks))

        else:
            tcs_df = pd.DataFrame([[gen_lat,gen_lon,month_seed,year]],columns=['lat','lon','month','year'])
            tcs_df_list.append(tcs_df)

    _log('year=%d done in %.1fs (setup %.1fs + tracks %.1fs)' %
         (year, time.time() - t0, t_tracks - t0, time.time() - t_tracks))
    seed_tries = pd.concat(seeds_df_list)
    tc_tries = pd.concat(tcs_df_list)

    return((tc_lon, tc_lat, tc_v, tc_m, tc_vmax, tc_rmax, tc_r34, tc_env_wnds, tc_month, tc_basin, n_seeds, seed_tries, tc_tries)) 

"""
Runs the downscaling model in basin "basin_id" according to the
settings in the namelist.txt file.
"""
def run_downscaling(basin_id):
    n_tracks = namelist.tracks_per_year   # number of tracks per year
    n_procs = namelist.n_procs
    b = basins.TC_Basin(basin_id)
    yearS = namelist.start_year
    yearE = namelist.end_year

    s = time.time()

    n_years = yearE - yearS + 1

    basin_ids = sorted([k for k in namelist.basin_bounds if k != 'GL'])
    save_yearly = bool(getattr(namelist, 'save_yearly', False))
    os.makedirs('%s/%s' % (namelist.output_directory, namelist.exp_name), exist_ok=True)

    if save_yearly:
        all_years = list(range(yearS, yearE + 1))
        todo_years = [yr for yr in all_years if not os.path.exists(get_fn_tracks_yearly(b, yr))]
        skipped = [yr for yr in all_years if yr not in todo_years]
        if skipped:
            print('[run_downscaling] save_yearly: skipping %d already done! year(s): %s' % (len(skipped), skipped), flush=True)
        if not todo_years:
            print('[run_downscaling] nothing to do, all yearly files exist', flush=True)
            return

        seed_csv = '%s/%s/seed_tries_%s.csv' % (namelist.output_directory, namelist.exp_name, namelist.exp_prefix)
        tc_csv = '%s/%s/tc_tries_%s.csv' % (namelist.output_directory, namelist.exp_name, namelist.exp_prefix)

        def _save_year(yr, out_tuple):
            ds_yr = _build_tracks_dataset([out_tuple], [yr], basin_ids)
            fn_yr = get_fn_tracks_yearly(b, yr)
            ds_yr.to_netcdf(fn_yr, mode='w')
            print('[run_downscaling] saved %s' % fn_yr, flush=True)
            # Append per-year seed/tc tries CSV (header only if file new)
            for df, csv_path in ((out_tuple[11], seed_csv), (out_tuple[12], tc_csv)):
                header = not os.path.exists(csv_path)
                df.to_csv(csv_path, mode='a', header=header, index=False)

        use_dask_eff = namelist.use_dask and len(todo_years) > 1
        if use_dask_eff:
            dask.config.set({'distributed.worker.profile.enabled': False})
            n_workers = min(namelist.n_procs, len(todo_years))
            cl_args = {'n_workers': n_workers, 'processes': True, 'threads_per_worker': 1}
            print('[run_downscaling] dask cluster: %d workers for %d year(s) (save_yearly=True)' % (n_workers, len(todo_years)), flush=True)
            with LocalCluster(**cl_args) as cluster, Client(cluster) as client:
                # Use futures + as_completed so we persist each year to disk
                futures = {client.submit(run_tracks, yr, n_tracks, b): yr for yr in todo_years}
                for fut in as_completed(futures):
                    yr = futures[fut]
                    try:
                        out_tuple = fut.result()
                    except Exception as err:
                        print('[run_downscaling] year %d FAILED: %r' % (yr, err), flush=True)
                        continue
                    _save_year(yr, out_tuple)
                    del out_tuple  # release worker memory
        else:
            if namelist.use_dask:
                print('[run_downscaling] only %d year(s); running serially' % len(todo_years), flush=True)
            for yr in todo_years:
                out_tuple = run_tracks(yr, n_tracks, b)
                _save_year(yr, out_tuple)

        print('[run_downscaling] done in %.1fs' % (time.time() - s), flush=True)
        return

    # Don't bother with dask overhead when there's nothing to parallelize over:
    use_dask_eff = namelist.use_dask and n_years > 1
    if use_dask_eff:
        dask.config.set({'distributed.worker.profile.enabled': False})
        n_workers = min(namelist.n_procs, n_years)
        cl_args = {'n_workers': n_workers,
                   'processes': True,
                   'threads_per_worker': 1}
        print('[run_downscaling] dask cluster: %d workers for %d years' %
              (n_workers, n_years), flush=True)
        with LocalCluster(**cl_args) as cluster, Client(cluster) as client:
            lazy_results = []
            f_args = []
            for yr in range(yearS, yearE+1):
                lazy_result = dask.delayed(run_tracks)(yr, n_tracks, b)
                f_args.append((yr, n_tracks, b))
                lazy_results.append(lazy_result)
            
            out = dask.compute(*lazy_results)
    else:
        if namelist.use_dask:
            print('[run_downscaling] only %d year(s); running serially '
                  '(dask overhead not worth it)' % n_years, flush=True)
        f_args = [(yr, n_tracks, b) for yr in range(yearS, yearE+1)]
        out = [run_tracks(yr, n_tracks, b) for yr in range(yearS, yearE+1)]


    seed_tries = pd.concat([x[11] for x in out], axis=0)
    tc_tries = pd.concat([x[12] for x in out], axis=0)

    yr_trks = np.stack([[x[0]] for x in f_args]).flatten()
    ds = _build_tracks_dataset(list(out), list(yr_trks), basin_ids)

    fn_trk_out = fn_tracks_duplicates(get_fn_tracks(b))

    seed_tries.to_csv('%s/%s/seed_tries_%s.csv' % (namelist.output_directory, namelist.exp_name, namelist.exp_prefix), mode='w')
    tc_tries.to_csv('%s/%s/tc_tries_%s.csv' % (namelist.output_directory, namelist.exp_name, namelist.exp_prefix), mode='w')

    ds.to_netcdf(fn_trk_out, mode='w')
    print('Saved %s' % fn_trk_out)

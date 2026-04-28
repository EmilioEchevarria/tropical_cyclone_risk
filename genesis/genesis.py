import dask
import numpy as np
import xarray as xr

import metpy.calc as mpcalc
import metpy.constants as mpconst
from metpy.units import units
from scipy.ndimage import uniform_filter
import namelist
from util import input

# Cache the Cd filter so we don't redo the load + interp + smoothing for every year.
_CD_FILTER_CACHE = {}

def _cd_filter(lons, lats):
    """
    Build the Cd-based topography filter on the requested (lons, lats) grid.
    The filter values are ~1 over open ocean, 0 over high-Cd land
    We multiplying TCGP by this filter to suppresses genesis probability 
    over terrain where TCs cannot physically spin up
    """

    tcgp_cd_filter = True
    tcgp_cd_path = '%s/intensity/data/Cd.nc' % namelist.src_directory
    tcgp_cd_clip = 0.025                   # max Cd before smoothing (caps highest peaks)
    tcgp_cd_ocean = 0.0007114              # nominal ocean Cd (filter = 1 here)
    tcgp_cd_smooth_size = 10               # uniform_filter size in grid cells

    lons = np.asarray(lons)
    lats = np.asarray(lats)
    key = (lons.shape, float(lons[0]), float(lons[-1]),
           lats.shape, float(lats[0]), float(lats[-1]))
    if key in _CD_FILTER_CACHE:
        return _CD_FILTER_CACHE[key]

    ds_cd = xr.open_dataset(tcgp_cd_path)
    # Auto-detect the lon/lat coord names in the Cd file.
    cd_lon = 'longitude' if 'longitude' in ds_cd.coords else 'lon'
    cd_lat = 'latitude' if 'latitude' in ds_cd.coords else 'lat'
    ds_cd = ds_cd.interp({cd_lon: lons, cd_lat: lats})
    
    cd = np.array(ds_cd['Cd'])
    cd = np.where(np.isnan(cd), tcgp_cd_ocean, cd)
    cd = np.minimum(cd, tcgp_cd_clip)

    cd_ocean = tcgp_cd_ocean
    cd_smoothed = uniform_filter(cd, size=tcgp_cd_smooth_size)

    # Renormalise smoothed field back to [cd_ocean, cd.max()]
    sm_min = cd_smoothed.min()
    sm_max = cd_smoothed.max()
    if sm_max > sm_min:
        cd_smoothed = (cd_ocean + (cd_smoothed-sm_min)/(sm_max-sm_min) * (cd.max()-cd_ocean))
    else:
        cd_smoothed = np.full_like(cd_smoothed, cd_ocean)

    denom = cd.max() - cd_ocean
    if denom <= 0:
        filt = np.ones_like(cd_smoothed)
    else:
        filt = 1.0 - (cd_smoothed - cd_ocean) / denom
    filt = np.clip(filt, 0.0, 1.0).astype(np.float32)

    _CD_FILTER_CACHE[key] = filt
    return filt


def _tcgp(vpot: xr.DataArray, xi: xr.DataArray, rh: xr.DataArray, shear: xr.DataArray):
    """
    TC genesis parameter calculation, based in part on Tory et al. (2018)
    doi: 10.1007/s00382-017-3752-4

    :param vpot: potential intensity
    :param xi: normalised vorticity parameter
    :param rh: mid-level relative humidity (700 hPa)
    :param shear: 200-850 hPa wind shear
    :return: `xr.DataArray` of TC genesis parameter
    """

    nu = xr.where(((vpot / 40) - 1) < 0, 0, (vpot / 40) - 1)
    mu = xr.where(((xi / 2e-5) - 1) < 0, 0, (xi / 2e-5) - 1)
    rho = xr.where(((rh*100 / 40) - 1) < 0, 0, (rh*100 / 40) - 1)
    sigma = xr.where((1 - (shear / 20)) < 0, 0, 1 - (shear / 20))

    tcgp = nu * mu * rho * sigma

    # Apply Cd filtering:
    lon_ky = input.get_lon_key() if input.get_lon_key() in tcgp.coords else 'lon'
    lat_ky = input.get_lat_key() if input.get_lat_key() in tcgp.coords else 'lat'
    filt = _cd_filter(tcgp[lon_ky].values, tcgp[lat_ky].values)
    tcgp = tcgp * xr.DataArray(
        filt, dims=(lat_ky, lon_ky),
        coords={lat_ky: tcgp[lat_ky], lon_ky: tcgp[lon_ky]})

    return tcgp.metpy.dequantify()


def _xi(ds: xr.Dataset):
    r"""
    Calculate ratio of 850-hPa absolute vorticity to normalised gradient of
    700 hPa absolute vorticity.

    .. math::
        \xi = \frac{|\eta_{850}|}{\beta_{*}(R / 2\Omega)}

    where:
    - :math:`\eta_{850}` is the 850 hPa absolute vorticity
    - :math:`\beta_{*}` is the meridional gradient of 700 hPa absolute vorticity
    - :math:`R` is the radius of the Earth
    - :math:`\Omega` is the rotational frequency of the Earth

    :param ds: `xr.Dataset` containing monthly mean wind data at 850, 700 hPa
    :return:  `xr.Dataarray` containing values of vorticity ratio :math:`\xi`
    :rtype: xr.DataArray
    """

    R = mpconst.earth_avg_radius
    omega = mpconst.earth_avg_angular_vel
    ua700 = ds['ua700_Mean'] * units('m/s')
    va700 = ds['va700_Mean'] * units('m/s')
    ua850 = ds['ua850_Mean'] * units('m/s')
    va850 = ds['va850_Mean'] * units('m/s')

    avrt700 = mpcalc.absolute_vorticity(ua700, va700)
    dedy, _ = mpcalc.gradient(
        avrt700, axes=[
            input.get_lat_key(),
            input.get_lon_key()]
    )
    #beta_floor = 5e-12 / (units.meter * units.second)
    #beta = xr.where(dedy < beta_floor, beta_floor, dedy)
    beta_floor = xr.full_like(dedy, 5e-12) * (1 / (units.meter * units.second))
    beta = xr.where(dedy < beta_floor, beta_floor, dedy)
    avrt850 = mpcalc.absolute_vorticity(ua850, va850)
    xi = np.abs(avrt850) / (beta * (R/(2 * omega)))
    xi = mpcalc.smooth_n_point(xi, 9, 2)
    return xi.metpy.dequantify()


def _shear(ds: xr.Dataset):
    """
    Calculate magnitude of vertical wind shear

    :param ds: _description_
    :type ds: _type_
    :return: _description_
    :rtype: _type_
    """
    ua250 = ds['ua250_Mean'] * units('m/s')
    va250 = ds['va250_Mean'] * units('m/s')
    ua850 = ds['ua850_Mean'] * units('m/s')
    va850 = ds['va850_Mean'] * units('m/s')

    shear = np.sqrt((ua250 - ua850)**2 + (va250 - va850)**2)
    shear = mpcalc.smooth_n_point(shear, 9, 2)
    return shear.metpy.dequantify()
import dask
import numpy as np
import xarray as xr

import metpy.calc as mpcalc
import metpy.constants as mpconst
from metpy.units import units

from util import input


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
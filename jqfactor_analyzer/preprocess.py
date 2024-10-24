# encoding: utf-8

import warnings

import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize as spwinsorize
from decimal import Decimal
from .utils import ignore_warning


def winsorize(data, scale=None, range=None, qrange=None, inclusive=True, inf2nan=True, axis=1):

    if isinstance(data, pd.DataFrame):
        return data.apply(
            winsorize,
            axis,
            scale=scale,
            range=range,
            qrange=qrange,
            inclusive=inclusive,
            inf2nan=inf2nan
        )
    elif (isinstance(data, np.ndarray) and data.ndim > 1):
        return np.apply_along_axis(
            winsorize,
            axis,
            arr=data,
            scale=scale,
            range=range,
            qrange=qrange,
            inclusive=inclusive,
            inf2nan=inf2nan
        )

    if isinstance(data, pd.Series):
        v = data.values
    else:
        v = data

    if not np.isfinite(v).any():
        return data

    # 如果v是int arrary，无法给 array 赋值 np.nan，因为 np.nan 是个 float
    v = v.astype(float)

    if inf2nan:
        v[~np.isfinite(v)] = np.nan

    if qrange:
        if (0 <= qrange[0] <= 1) and (0 <= qrange[1] <= 1):
            raise Exception(u'qrange 值应在 0 到 1 之间，如 [0.05, 0.95]')
        qrange = (Decimal(str(qrange[0])), 1 - Decimal(str(qrange[1])))

        if inclusive:
            v[~np.isnan(v)] = spwinsorize(v[~np.isnan(v)], qrange, inclusive=[True, True])
        else:
            # 如果v是int arrary，无法给 array 赋值 np.nan，因为 np.nan 是个 float
            v = v.astype(float)
            not_nan = v[~np.isnan(v)]
            not_nan[not_nan != spwinsorize(not_nan, qrange, inclusive=[True, True])] = np.nan
            v[~np.isnan(v)] = not_nan

    else:
        if range:
            range_ = (Decimal(str(range[0])) if not np.isnan(range[0]) else np.nan,
                      Decimal(str(range[1])) if not np.isnan(range[1]) else np.nan)
        else:
            mu = np.mean(data[np.isfinite(data)])
            sigma = np.std(data[np.isfinite(data)])
            range_ = (np.nanmin(v[v > mu - scale * sigma]),
                      np.nanmax(v[v < mu + scale * sigma]))

        if inclusive:
            not_nan = ~np.isnan(v)
            v[not_nan] = np.where(v[not_nan] < range_[0], range_[0], v[not_nan])
            not_nan = ~np.isnan(v)
            v[not_nan] = np.where(v[not_nan] > range_[1], range_[1], v[not_nan])
        else:
            not_nan = ~np.isnan(v)
            v_not_nan = v[not_nan]
            v[not_nan] = np.where(
                np.logical_and(v_not_nan >= range_[0], v_not_nan <= range_[1]), v_not_nan, np.nan
            )

    if isinstance(data, pd.Series):
        return pd.Series(v, index=data.index)
    else:
        return v


def winsorize_med(data, scale=1, inclusive=True, inf2nan=True, axis=1):

    if isinstance(data, pd.DataFrame):
        return data.apply(winsorize_med, axis, scale=scale, inclusive=inclusive, inf2nan=inf2nan)
    elif (isinstance(data, np.ndarray) and data.ndim > 1):
        return np.apply_along_axis(
            winsorize_med, axis, arr=data, scale=scale, inclusive=inclusive, inf2nan=inf2nan
        )

    if isinstance(data, pd.Series):
        v = data.values
    else:
        v = data

    if not np.isfinite(v).any():
        return data

    # 如果v是int arrary，无法给 array 赋值 np.nan，因为 np.nan 是个 float
    v = v.astype(float)

    if inf2nan:
        v[~np.isfinite(v)] = np.nan

    med = np.median(v[~np.isnan(v)])

    data_minus_med = v[~np.isnan(v)] - med
    median_absolute = np.median(np.abs(data_minus_med))

    if inclusive:
        not_nan = ~np.isnan(v)
        v[not_nan] = np.where(
            v[not_nan] > med + scale * median_absolute, med + scale * median_absolute, v[not_nan]
        )
        not_nan = ~np.isnan(v)
        v[not_nan] = np.where(
            v[not_nan] < med - scale * median_absolute, med - scale * median_absolute, v[not_nan]
        )
    else:
        # 如果v是int arrary，np.nan 会被转换成一个极小的数，比如 -2147483648
        v = v.astype(float)
        not_nan = ~np.isnan(v)
        v_not_nan = v[not_nan]
        v[not_nan] = np.where(
            np.logical_and(
                v_not_nan <= med + scale * median_absolute,
                v_not_nan >= med - scale * median_absolute
            ), v_not_nan, np.nan
        )

    if isinstance(data, pd.Series):
        return pd.Series(v, index=data.index)
    else:
        return v


@ignore_warning(message='Mean of empty slice', category=RuntimeWarning)
@ignore_warning(message='Degrees of freedom <= 0 for slice',
                category=RuntimeWarning)
@ignore_warning(message='invalid value encountered in true_divide',
                category=RuntimeWarning)
def standardlize(data, inf2nan=True, axis=1):
    if inf2nan:
        data = data.astype('float64')
        data[np.isinf(data)] = np.nan

    axis = min(data.ndim - 1, axis)

    if not np.any(np.isfinite(data)):
        return data

    mu = np.nanmean(np.where(~np.isinf(data), data, np.nan), axis=axis)
    std = np.nanstd(np.where(~np.isinf(data), data, np.nan), axis=axis)

    rep = np.tile if axis == 0 else np.repeat
    mu = np.asarray(rep(mu, data.shape[axis])).reshape(data.shape)
    std = np.asarray(rep(std, data.shape[axis])).reshape(data.shape)

    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.where(np.isinf(data), (data - mu) / std)
    else:
        data = np.where(np.isinf(data), data, (data - mu) / std)
    return data

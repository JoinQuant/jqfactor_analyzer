# -*- coding: utf-8 -*-

"""pandas库版本兼容模块"""

import warnings

import pandas as pd


# pandas
PD_VERSION = pd.__version__


def rolling_apply(
    x,
    window,
    func,
    min_periods=None,
    freq=None,
    center=False,
    args=None,
    kwargs=None
):
    if args is None:
        args = tuple()
    if kwargs is None:
        kwargs = dict()

    if PD_VERSION >= '0.23.0':
        return x.rolling(
            window, min_periods=min_periods, center=center
        ).apply(
            func, False, args=args, kwargs=kwargs
        )
    elif PD_VERSION >= '0.18.0':
        return x.rolling(
            window, min_periods=min_periods, center=center
        ).apply(
            func, args=args, kwargs=kwargs
        )
    else:
        return pd.rolling_apply(
            x,
            window,
            func,
            min_periods=min_periods,
            freq=freq,
            center=center,
            args=args,
            kwargs=kwargs
        )


def rolling_mean(x, window, min_periods=None, center=False):
    if PD_VERSION >= '0.18.0':
        return x.rolling(window, min_periods=min_periods, center=center).mean()
    else:
        return pd.rolling_mean(
            x, window, min_periods=min_periods, center=center
        )


def rolling_std(x, window, min_periods=None, center=False, ddof=1):
    if PD_VERSION >= '0.18.0':
        return x.rolling(
            window, min_periods=min_periods, center=center
        ).std(ddof=ddof)
    else:
        return pd.rolling_std(
            x, window, min_periods=min_periods, center=center, ddof=ddof
        )

# -*- coding: utf-8 -*-


import re
import six
import warnings
from functools import wraps
from collections import Iterable

import pandas as pd


def get_forward_returns_columns(columns):
    syntax = re.compile("^period_\\d+$")
    return columns[columns.astype('str').str.contains(syntax, regex=True)]


def convert_to_forward_returns_columns(period):
    try:
        return 'period_{:d}'.format(period)
    except ValueError:
        return period


def ignore_warning(message='', category=Warning, module='', lineno=0, append=False):
    """过滤 warnings"""
    def decorator(func):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message=message, category=category,
                                        module=module, lineno=lineno, append=append)
                return func(*args, **kwargs)
        return func_wrapper

    return decorator


def ensure_tuple(x):
    if isinstance(x, six.string_types) or not isinstance(x, Iterable):
        return (x,)
    else:
        return tuple(x)

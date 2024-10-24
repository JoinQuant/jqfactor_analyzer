import os
import time
import shutil
import numpy as np
import pandas as pd

import jqdata
from jqfactor_analyzer.data import DataApi


def test_cache():
    # api1 不开启缓存, api2 开启缓存
    api1 = DataApi(weight_method='mktcap', allow_cache=False)
    api2 = DataApi(weight_method='mktcap')
    codes = jqdata.apis.get_all_securities('stock').index.tolist()
    start_date = '2024-07-01'
    end_date = '2024-07-10'

    df1 = api1.apis['weights'](codes, start_date, end_date)
    df2 = api2.apis['weights'](codes, start_date, end_date)
    assert df1.equals(df2)

    api1.weight_method = api2.weight_method = 'cmktcap'
    df1 = api1.apis['weights'](codes, start_date, end_date)
    df2 = api2.apis['weights'](codes, start_date, end_date)
    assert df1.equals(df2)

    df1 = api1.apis['prices'](codes, start_date, end_date)
    df2 = api2.apis['prices'](codes, start_date, end_date)
    assert df1.equals(df2)

    # 非后复权的 price 存在微量差异
    api1.fq = 'pre'
    api2.fq = 'pre'
    df1 = api1.apis['prices'](codes, start_date, end_date)
    df2 = api2.apis['prices'](codes, start_date, end_date)
    for code in codes:
        diff = (df1[code] - df2[code]).abs().sum()
        assert diff < 1e-12

    api1.fq = None
    api1.price = 'open'
    api2.fq = None
    api2.price = 'open'
    df1 = api1.apis['prices'](codes, start_date, end_date)
    df2 = api2.apis['prices'](codes, start_date, end_date)
    for code in codes:
        diff = (df1[code] - df2[code]).abs().sum()
        assert diff < 1e-12

    df1 = api1.apis['groupby'](codes, start_date, end_date)
    df2 = api2.apis['groupby'](codes, start_date, end_date)
    assert df1.equals(df2)

    cache_path = os.path.expanduser(api2.cfg["default_dir"])
    shutil.rmtree(cache_path)

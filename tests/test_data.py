import os
import shutil

from jqfactor_analyzer.data import DataApi
from jqfactor_analyzer.preprocess import *
from jqfactor_analyzer.factor_cache import *


try:
    import jqdata
except:
    import jqdatasdk
    # 使用 sdk 进行测试时需要先登陆
    # jqdatasdk.auth("ACCOUNT", "PASSWORD")


def test_preprocess():
    api = DataApi(weight_method='mktcap')
    codes = api._api.get_all_securities('stock').index.tolist()
    start_date = '2024-07-05'
    end_date = '2024-07-15'
    df = api.apis['prices'](codes, start_date, end_date).dropna(how='all', axis=1)

    w_df = winsorize(df, scale=1)
    assert all(df.max() >= w_df.max())

    wm_df = winsorize_med(df, scale=1)
    assert not wm_df.equals(w_df)

    s_df = standardlize(df)
    assert set(s_df.std(axis=1).round()) == {1.0}

    n_df = neutralize(df, how='sw_l3', date='2024-07-10')
    assert n_df.shape == (7, 5111)


def test_cache():
    # api1 不开启缓存, api2 开启缓存
    api1 = DataApi(weight_method='mktcap', allow_cache=False)
    api2 = DataApi(weight_method='mktcap')
    codes = api1._api.get_all_securities('stock').index.tolist()
    start_date = '2024-07-01'
    end_date = '2024-07-10'

    df1 = api1.apis['weights'](codes, start_date, end_date)
    df2 = api2.apis['weights'](codes, start_date, end_date)
    for code in codes:
        assert (df1[code] - df2[code]).abs().sum() < 1e-3

    api1.weight_method = api2.weight_method = 'cmktcap'
    df1 = api1.apis['weights'](codes, start_date, end_date)
    df2 = api2.apis['weights'](codes, start_date, end_date)
    for code in codes:
        assert (df1[code] - df2[code]).abs().sum() < 1e-3

    df1 = api1.apis['prices'](codes, start_date, end_date)
    df2 = api2.apis['prices'](codes, start_date, end_date)
    assert df1.equals(df2)

    # 非后复权的 price 存在微量差异
    api1.fq = 'pre'
    api2.fq = 'pre'
    df1 = api1.apis['prices'](codes, start_date, end_date)  # 无缓存
    df2 = api2.apis['prices'](codes, start_date, end_date)  # 有缓存
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

    # 删除缓存文件
    cache_path = get_cache_dir()
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)

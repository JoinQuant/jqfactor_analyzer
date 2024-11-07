# encoding: utf-8

import warnings

import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize as spwinsorize
from decimal import Decimal
from .utils import ignore_warning

from .data import  DataApi,convert_date
from fastcache import lru_cache
from functools import partial
from statsmodels.api import OLS, add_constant as sm_add_constant



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
        if not ((0 <= qrange[0] <= 1) and (0 <= qrange[1] <= 1)):
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


@lru_cache(3)
def cache_dataapi(allow_cache=True, show_progress=False):
    return DataApi(allow_cache=allow_cache, show_progress=show_progress)


def get_neu_basicdata(how, securities, date=None):
    """获取中性化的依赖数据
    返回: 一个 DataFrame, index 是股票代码
    """
    if isinstance(how, str):
        how = [how]

    if isinstance(how, (pd.Series, pd.DataFrame)):
        return how
    elif isinstance(how, (list, tuple)):
        how_datas = []
    else:
        raise ValueError("错误的 how 参数格式 : {}".format(how))

    dataapi = cache_dataapi()
    for how_name in how:
        if isinstance(how_name, pd.Series):
            how_datas.append(how_name.to_frame())
        elif isinstance(how_name, pd.DataFrame):
            how_datas.append(how_name)
        elif how_name in ['jq_l1', 'jq_l2', 'sw_l1', 'sw_l2', 'sw_l3', 'zjw']:
            industry_info = pd.get_dummies(dataapi._get_cached_industry_one_day(
                date, securities, industry=how_name)).reindex(securities, fill_value=0)
            how_datas.append(industry_info)
        elif how_name in ['mktcap', 'ln_mktcap', 'cmktcap', 'ln_cmktcap']:
            if how_name == 'mktcap':
                mkt_api = partial(dataapi._get_market_cap, ln=False)
            elif how_name == 'ln_mktcap':
                mkt_api = partial(dataapi._get_market_cap, ln=True)
            elif how_name == 'cmktcap':
                mkt_api = partial(dataapi._get_circulating_market_cap, ln=False)
            elif how_name == 'ln_cmktcap':
                mkt_api = partial(dataapi._get_circulating_market_cap, ln=True)

            market_info= mkt_api(securities=securities, start_date=date, end_date=date).T
            market_info.columns=[how_name]
            how_datas.append(market_info)
        else:
            raise ValueError("不支持的因子名称 : {} ".format(how_name))

    return pd.concat(how_datas,axis=1)


def neutralize(data, how=None, date=None, axis=1, fillna=None, add_constant=False):
    """中性化
    data: pd.Series/pd.DataFrame, 待中性化的序列, 序列的 index/columns 为股票的 code
    how: str list. 中性化使用的因子名称列表. 默认为 ['jq_l1', 'market_cap'], 支持的中性化方法有:
                1. 行业: sw_l1, sw_l2, sw_l3, jq_l1, jq_l2
                2. 市值因子: mktcap(总市值), ln_mktcap(对数总市值), cmktcap(流通市值), ln_cmktcap(对数流通市值)
                3. 自定义的中性化数据: 支持同时传入额外的 Series 或者 DataFrame 用来进行中性化, index 必须是标的代码
                以上三类参数可同时传入参数列表
    date: 日期, 将用 date 这天的相关变量数据对 series 进行中性化 (注意依赖数据的实际可用时间, 如市值数据当天盘中是无法获取到的)
    axis: 默认为 1. 仅在 data 为 pd.DataFrame 时生效. 表示沿哪个方向做中性化, 0 为对每列做中性化, 1 为对每行做中性化
    fillna: 缺失值填充方式, 默认为None, 表示不填充. 支持的值:
        'jq_l1': 聚宽一级行业
        'jq_l2': 聚宽二级行业
        'sw_l1': 申万一级行业
        'sw_l2': 申万二级行业
        'sw_l3': 申万三级行业 表示使用某行业分类的均值进行填充.
    add_constant: 中性化时是否添加常数项, 默认为 False
    """
    if data.dropna(how='all').empty:
        return data

    if how is None:
        how = ['jq_l1', 'mktcap']
    elif isinstance(how, str):
        how = [how]

    if isinstance(data, pd.Series) or axis == 0:
        securities = data.index.astype(str)
    else:
        securities = data.columns.astype(str)
    invalid_securities = securities[~(securities.str.endswith("XSHG") | securities.str.endswith("XSHE"))].tolist()
    if invalid_securities:
        raise ValueError('neutralize: 找不到股票: {sym:s}'.format(sym=str(invalid_securities)))

    exposure = get_neu_basicdata(how, securities.tolist(), date=date)

    with pd.option_context('mode.use_inf_as_null', True):
        exposure.dropna(axis=1, how='all', inplace=True)
        exposure.dropna(inplace=True)
        exposure = exposure.astype(np.float64)

    if exposure.empty:
        return data

    if fillna is not None:
        dataapi = cache_dataapi()
        ind = dataapi._get_cached_industry_one_day(date, securities)

    def valid_index(s):
        return s[np.isfinite(s)].index.intersection(exposure.index)

    def get_resid(s):
        valid_index_ = valid_index(s)
        if len(valid_index_) > 1:
            resid = OLS(
                s.loc[valid_index_].values,
                (sm_add_constant(exposure.loc[valid_index_].values) if add_constant
                 else exposure.loc[valid_index_].values),
                missing='drop'
            ).fit().resid
            resid = pd.Series(resid, index=valid_index_)
            resid = resid.reindex(s.index, fill_value=np.nan)
            if fillna is not None:
                resid = resid.groupby(ind.loc[s.index]).apply(lambda x: x.fillna(x.mean()))
        else:
            resid = pd.Series(np.nan, index=s.index)
        return resid

    if isinstance(data, pd.Series):
        return get_resid(data)
    else:
        return data.apply(get_resid, axis)


__all__ = [
    'neutralize',
    'winsorize',
    'winsorize_med',
    'standardlize',
]

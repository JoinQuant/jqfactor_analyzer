# -*- coding: utf-8 -*-


from __future__ import division

import pandas as pd
import numpy as np

from .exceptions import MaxLossExceededError, non_unique_bin_edges_error
from .utils import get_forward_returns_columns


@non_unique_bin_edges_error
def quantize_factor(
    factor_data, quantiles=5, bins=None, by_group=False, no_raise=False, zero_aware=False,
):
    """
    计算每期因子分位数

    参数
    ----------
    factor_data : pd.DataFrame - MultiIndex
        一个 DataFrame, index 为日期 (level 0) 和资产(level 1) 的 MultiIndex,
        values 包括因子的值, 各期因子远期收益, 因子分位数,
        因子分组(可选), 因子权重(可选)
    quantiles : int or sequence[float]
        在因子分组中按照因子值大小平均分组的组数。
         或分位数序列, 允许不均匀分组
        例如 [0, .10, .5, .90, 1.] 或 [.05, .5, .95]
        'quantiles' 和 'bins' 有且只能有一个不为 None
    bins : int or sequence[float]
        在因子分组中使用的等宽 (按照因子值) 区间的数量
        或边界值序列, 允许不均匀的区间宽度
        例如 [-4, -2, -0.5, 0, 10]
        'quantiles' 和 'bins' 有且只能有一个不为 None
    by_group : bool
        如果是 True, 按照 group 分别计算分位数
    no_raise: bool, optional
        如果为 True，则不抛出任何异常，并且将抛出异常的值设置为 np.NaN
    zero_aware : bool, optional
        如果为True，则分别为正负因子值计算分位数。
        适用于您的信号聚集并且零是正值和负值的分界线的情况.

    返回值
    -------
    factor_quantile : pd.Series
        index 为日期 (level 0) 和资产(level 1) 的因子分位数
    """
    if not ((quantiles is not None and bins is None) or
            (quantiles is None and bins is not None)):
        raise ValueError('quantiles 和 bins 至少要输入一个')

    if zero_aware and not (isinstance(quantiles, int)
                           or isinstance(bins, int)):
        msg = ("只有 quantiles 或 bins 为 int 类型时， 'zero_aware' 才能为 True")
        raise ValueError(msg)

    def quantile_calc(x, _quantiles, _bins, _zero_aware, _no_raise):
        try:
            if _quantiles is not None and _bins is None and not _zero_aware:
                return pd.qcut(x, _quantiles, labels=False) + 1
            elif _quantiles is not None and _bins is None and _zero_aware:
                pos_quantiles = pd.qcut(x[x >= 0], _quantiles // 2,
                                        labels=False) + _quantiles // 2 + 1
                neg_quantiles = pd.qcut(x[x < 0], _quantiles // 2,
                                        labels=False) + 1
                return pd.concat([pos_quantiles, neg_quantiles]).sort_index()
            elif _bins is not None and _quantiles is None and not _zero_aware:
                return pd.cut(x, _bins, labels=False) + 1
            elif _bins is not None and _quantiles is None and _zero_aware:
                pos_bins = pd.cut(x[x >= 0], _bins // 2,
                                  labels=False) + _bins // 2 + 1
                neg_bins = pd.cut(x[x < 0], _bins // 2,
                                  labels=False) + 1
                return pd.concat([pos_bins, neg_bins]).sort_index()
        except Exception as e:
            if _no_raise:
                return pd.Series(index=x.index)
            raise e

    grouper = [factor_data.index.get_level_values('date')]
    if by_group:
        if 'group' not in factor_data.columns:
            raise ValueError('只有输入了 groupby 参数时 binning_by_group 才能为 True')
        grouper.append('group')

    factor_quantile = factor_data.groupby(grouper)['factor'] \
        .apply(quantile_calc, quantiles, bins, zero_aware, no_raise)
    factor_quantile.name = 'factor_quantile'

    return factor_quantile.dropna()


def compute_forward_returns(factor,
                            prices,
                            periods=(1, 5, 10)):
    """
    计算每个因子值对应的 N 期因子远期收益

    参数
    ----------
    factor : pd.Series - MultiIndex
        一个 Series, index 为日期 (level 0) 和资产(level 1) 的 MultiIndex,
        values 为因子值
    prices : pd.DataFrame
        用于计算因子远期收益的价格数据
        columns 为资产, index 为 日期.
        价格数据必须覆盖因子分析时间段以及额外远期收益计算中的最大预期期数.
    periods : sequence[int]
        远期收益的期数
    Returns
    -------
    forward_returns : pd.DataFrame - MultiIndex
        因子远期收益
        index 为日期 (level 0) 和资产(level 1) 的 MultiIndex
        column 为远期收益的期数
    """

    factor_dateindex = factor.index.levels[0]
    factor_dateindex = factor_dateindex.intersection(prices.index)

    if len(factor_dateindex) == 0:
        raise ValueError("Factor and prices indices don't match: make sure "
                         "they have the same convention in terms of datetimes "
                         "and symbol-names")

    prices = prices.filter(items=factor.index.levels[1])

    forward_returns = pd.DataFrame(
        index=pd.MultiIndex
        .from_product([prices.index, prices.columns], names=['date', 'asset'])
    )

    for period in periods:
        delta = prices.pct_change(period).shift(-period).reindex(factor_dateindex)
        forward_returns['period_{p}'.format(p=period)] = delta.stack()

    forward_returns.index = forward_returns.index.rename(['date', 'asset'])

    return forward_returns


def demean_forward_returns(factor_data, grouper=None):
    """
    根据相关分组为因子远期收益去均值.
    分组去均值包含了投资组合分组中性化约束的假设，因此允许跨组评估因子.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        因子远期收益
        index 为日期 (level 0) 和资产(level 1) 的 MultiIndex
        column 为远期收益的期数
    grouper : list
        如果为 None, 则只根据日期去均值
        否则则根据列表中提供的组分组去均值

    返回值
    -------
    adjusted_forward_returns : pd.DataFrame - MultiIndex
        和 factor_data 相同形状的 DataFrame, 但每个收益都被分组去均值了
    """

    factor_data = factor_data.copy()

    if not grouper:
        grouper = factor_data.index.get_level_values('date')

    cols = get_forward_returns_columns(factor_data.columns)
    factor_data[cols] = factor_data.groupby(
        grouper, as_index=False
    )[cols.append(pd.Index(['weights']))].apply(
        lambda x: x[cols].subtract(
            np.average(x[cols], axis=0, weights=x['weights'].fillna(0.0).values),
            axis=1
        )
    )

    return factor_data


def get_clean_factor(factor,
                     forward_returns,
                     groupby=None,
                     weights=None,
                     binning_by_group=False,
                     quantiles=5,
                     bins=None,
                     max_loss=0.35,
                     zero_aware=False):
    """
    将因子值, 因子远期收益, 因子分组数据, 因子权重数据
    格式化为以时间和资产的 MultiIndex 作为索引的 DataFrame.

    参数
    ----------
    factor : pd.Series - MultiIndex
        一个 Series, index 为日期 (level 0) 和资产(level 1) 的 MultiIndex,
        values 为因子的值
    forward_returns : pd.DataFrame - MultiIndex
        一个 DataFrame, index 为日期 (level 0) 和资产(level 1) 的 MultiIndex,
        values 为因子的远期收益, columns 为因子远期收益的期数.
    groupby : pd.Series - MultiIndex or dict
        index 为日期和资产的 Series，为每个资产每天的分组，或资产-分组映射的字典.
        如果传递了dict，则假定分组映射在整个时间段内保持不变.
    weights : pd.Series - MultiIndex or dict
        index 为日期和资产的 Series，为每个资产每天的权重，或资产-权重映射的字典.
        如果传递了dict，则假定权重映射在整个时间段内保持不变.
    binning_by_group : bool
        如果为 True, 则对每个组分别计算分位数.
        适用于因子值范围在各个组上变化很大的情况.
        如果要分析分组(行业)中性的组合, 您最好设置为 True
    quantiles : int or sequence[float]
        在因子分组中按照因子值大小平均分组的组数。
         或分位数序列, 允许不均匀分组
        例如 [0, .10, .5, .90, 1.] 或 [.05, .5, .95]
        'quantiles' 和 'bins' 有且只能有一个不为 None
    bins : int or sequence[float]
        在因子分组中使用的等宽 (按照因子值) 区间的数量
        或边界值序列, 允许不均匀的区间宽度
        例如 [-4, -2, -0.5, 0, 10]
        'quantiles' 和 'bins' 有且只能有一个不为 None
    max_loss : float, optional
        允许的丢弃因子数据的最大百分比 (0.00 到 1.00),
        计算比较输入因子索引中的项目数和输出 DataFrame 索引中的项目数.
        因子数据本身存在缺陷 (例如 NaN),
        没有提供足够的价格数据来计算所有因子值的远期收益，
        或者因为分组失败, 因此可以部分地丢弃因子数据
        设置 max_loss = 0 以停止异常捕获.
    zero_aware : bool, optional
        如果为True，则分别为正负因子值计算分位数。
        适用于您的信号聚集并且零是正值和负值的分界线的情况.

    返回值
    -------
    merged_data : pd.DataFrame - MultiIndex
        一个 DataFrame, index 为日期 (level 0) 和资产(level 1) 的 MultiIndex,
        values 包括因子的值, 各期因子远期收益, 因子分位数,
        因子分组(可选), 因子权重(可选)
        - 各期因子远期收益的列名满足 'period_1', 'period_5' 的格式
    """

    initial_amount = float(len(factor.index))

    factor_copy = factor.copy()
    factor_copy.index = factor_copy.index.rename(['date', 'asset'])

    merged_data = forward_returns.copy()
    merged_data['factor'] = factor_copy

    if groupby is not None:
        if isinstance(groupby, dict):
            diff = set(factor_copy.index.get_level_values(
                'asset')) - set(groupby.keys())
            if len(diff) > 0:
                raise KeyError(
                    "Assets {} not in group mapping".format(
                        list(diff)))

            ss = pd.Series(groupby)
            groupby = pd.Series(index=factor_copy.index,
                                data=ss[factor_copy.index.get_level_values(
                                    'asset')].values)
        elif isinstance(groupby, pd.DataFrame):
            groupby = groupby.stack()
        merged_data['group'] = groupby

    if weights is not None:
        if isinstance(weights, dict):
            diff = set(factor_copy.index.get_level_values(
                'asset')) - set(weights.keys())
            if len(diff) > 0:
                raise KeyError(
                    "Assets {} not in weights mapping".format(
                        list(diff)))

            ww = pd.Series(weights)
            weights = pd.Series(index=factor_copy.index,
                                data=ww[factor_copy.index.get_level_values(
                                    'asset')].values)
        elif isinstance(weights, pd.DataFrame):
            weights = weights.stack()
        merged_data['weights'] = weights

    merged_data = merged_data.dropna()

    quantile_data = quantize_factor(
        merged_data,
        quantiles,
        bins,
        binning_by_group,
        True,
        zero_aware
    )

    merged_data['factor_quantile'] = quantile_data
    merged_data = merged_data.dropna()
    merged_data['factor_quantile'] = merged_data['factor_quantile'].astype(int)

    if 'weights' in merged_data.columns:
        merged_data['weights'] = merged_data.set_index(
            'factor_quantile', append=True
        ).groupby(level=['date', 'factor_quantile'])['weights'].apply(
            lambda s: s.divide(s.sum())
        ).reset_index('factor_quantile', drop=True)

    binning_amount = float(len(merged_data.index))

    tot_loss = (initial_amount - binning_amount) / initial_amount

    no_raise = True if max_loss == 0 else False
    if tot_loss > max_loss and not no_raise:
        message = ("max_loss (%.1f%%) 超过 %.1f%%"
                   % (tot_loss * 100, max_loss * 100))
        raise MaxLossExceededError(message)

    return merged_data


def get_clean_factor_and_forward_returns(factor,
                                         prices,
                                         groupby=None,
                                         weights=None,
                                         binning_by_group=False,
                                         quantiles=5,
                                         bins=None,
                                         periods=(1, 5, 10),
                                         max_loss=0.35,
                                         zero_aware=False):
    """
    将因子数据, 价格数据, 分组映射和权重映射格式化为
    由包含时间和资产的 MultiIndex 作为索引的 DataFrame

    参数
    ----------
    factor : pd.Series - MultiIndex
     一个 Series, index 为日期 (level 0) 和资产(level 1) 的 MultiIndex,
        values 为因子的值
    prices : pd.DataFrame
        用于计算因子远期收益的价格数据
        columns 为资产, index 为 日期.
        价格数据必须覆盖因子分析时间段以及额外远期收益计算中的最大预期期数.
    groupby : pd.Series - MultiIndex or dict
        index 为日期和资产的 Series，为每个资产每天的分组，或资产-分组映射的字典.
        如果传递了dict，则假定分组映射在整个时间段内保持不变.
    weights : pd.Series - MultiIndex or dict
        index 为日期和资产的 Series，为每个资产每天的权重，或资产-权重映射的字典.
        如果传递了dict，则假定权重映射在整个时间段内保持不变.
    binning_by_group : bool
        如果为 True, 则对每个组分别计算分位数.
        适用于因子值范围在各个组上变化很大的情况.
        如果要分析分组(行业)中性的组合, 您最好设置为 True
    quantiles : int or sequence[float]
        在因子分组中按照因子值大小平均分组的组数。
         或分位数序列, 允许不均匀分组
        例如 [0, .10, .5, .90, 1.] 或 [.05, .5, .95]
        'quantiles' 和 'bins' 有且只能有一个不为 None
    bins : int or sequence[float]
        在因子分组中使用的等宽 (按照因子值) 区间的数量
        或边界值序列, 允许不均匀的区间宽度
        例如 [-4, -2, -0.5, 0, 10]
        'quantiles' 和 'bins' 有且只能有一个不为 None
    periods : sequence[int]
        远期收益的期数
    max_loss : float, optional
        允许的丢弃因子数据的最大百分比 (0.00 到 1.00),
        计算比较输入因子索引中的项目数和输出 DataFrame 索引中的项目数.
        因子数据本身存在缺陷 (例如 NaN),
        没有提供足够的价格数据来计算所有因子值的远期收益，
        或者因为分组失败, 因此可以部分地丢弃因子数据
        设置 max_loss = 0 以停止异常捕获.
    zero_aware : bool, optional
        如果为True，则分别为正负因子值计算分位数。
        适用于您的信号聚集并且零是正值和负值的分界线的情况.

    返回值
    -------
    merged_data : pd.DataFrame - MultiIndex
        一个 DataFrame, index 为日期 (level 0) 和资产(level 1) 的 MultiIndex,
        values 包括因子的值, 各期因子远期收益, 因子分位数,
        因子分组(可选), 因子权重(可选)
        - 各期因子远期收益的列名满足 'period_1', 'period_5' 的格式
    """

    forward_returns = compute_forward_returns(factor, prices, periods)

    factor_data = get_clean_factor(factor, forward_returns, groupby=groupby,
                                   weights=weights,
                                   quantiles=quantiles, bins=bins,
                                   binning_by_group=binning_by_group,
                                   max_loss=max_loss, zero_aware=zero_aware)

    return factor_data


def common_start_returns(
    factor,
    prices,
    before,
    after,
    cumulative=False,
    mean_by_date=False,
    demean_by=None
):

    if cumulative:
        returns = prices
    else:
        returns = prices.pct_change(axis=0)

    all_returns = []

    for timestamp, df in factor.groupby(level='date'):

        equities = df.index.get_level_values('asset')

        try:
            day_zero_index = returns.index.get_loc(timestamp)
        except KeyError:
            continue

        starting_index = max(day_zero_index - before, 0)
        ending_index = min(day_zero_index + after + 1, len(returns.index))

        equities_slice = set(equities)
        if demean_by is not None:
            demean_equities = demean_by.loc[timestamp] \
                .index.get_level_values('asset')
            equities_slice |= set(demean_equities)

        series = returns.loc[returns.
                             index[starting_index:ending_index], equities_slice]
        series.index = range(
            starting_index - day_zero_index, ending_index - day_zero_index
        )

        if cumulative:
            series = (series / series.loc[0, :]) - 1

        if demean_by is not None:
            mean = series.loc[:, demean_equities].mean(axis=1)
            series = series.loc[:, equities]
            series = series.sub(mean, axis=0)

        if mean_by_date:
            series = series.mean(axis=1)

        all_returns.append(series)

    return pd.concat(all_returns, axis=1)


def rate_of_return(period_ret):
    """
    转换回报率为"每期"回报率：如果收益以稳定的速度增长, 则相当于每期的回报率
    """
    period = int(period_ret.name.replace('period_', ''))
    return period_ret.add(1).pow(1. / period).sub(1)


def std_conversion(period_std):
    """
    转换回报率标准差为"每期"回报率标准差
    """
    period_len = int(period_std.name.replace('period_', ''))
    return period_std / np.sqrt(period_len)

# -*- coding: utf-8 -*-

from __future__ import division, print_function

from collections import Iterable

import numpy as np
import pandas as pd
from fastcache import lru_cache
from cached_property import cached_property
from scipy.stats import spearmanr, pearsonr, morestats

from . import performance as pef, plotting as pl
from .prepare import (get_clean_factor_and_forward_returns, rate_of_return,
                      std_conversion)
from .plot_utils import _use_chinese
from .utils import convert_to_forward_returns_columns, ensure_tuple


class FactorAnalyzer(object):
    """单因子分析

        参数
        ----------
        factor :
            因子值
            pd.DataFrame / pd.Series
            一个 DataFrame, index 为日期, columns 为资产,
            values 为因子的值
            或一个 Series, index 为日期和资产的 MultiIndex,
            values 为因子的值
        prices :
            用于计算因子远期收益的价格数据
            pd.DataFrame
            index 为日期, columns 为资产
            价格数据必须覆盖因子分析时间段以及额外远期收益计算中的最大预期期数.
            或 function
            输入参数为 securities, start_date, end_date, count
            返回值为价格数据的 DataFrame
        groupby :
            分组数据, 默认为 None
            pd.DataFrame
            index 为日期, columns 为资产，为每个资产每天的分组.
            或 dict
            资产-分组映射的字典. 如果传递了dict，则假定分组映射在整个时间段内保持不变.
            或 function
            输入参数为 securities, start_date, end_date
            返回值为分组数据的 DataFrame 或 dict
        weights :
            权重数据, 默认为 1
            pd.DataFrame
            index 为日期, columns 为资产，为每个资产每天的权重.
            或 dict
            资产-权重映射的字典. 如果传递了dict，则假定权重映射在整个时间段内保持不变.
            或 function
            输入参数为 securities, start_date, end_date
            返回值为权重数据的 DataFrame 或 dict
        binning_by_group :
            bool
            如果为 True, 则对每个组分别计算分位数. 默认为 False
            适用于因子值范围在各个组上变化很大的情况.
            如果要分析分组(行业)中性的组合, 您最好设置为 True
        quantiles :
            int or sequence[float]
            默认为 None
            在因子分组中按照因子值大小平均分组的组数
            或分位数序列, 允许不均匀分组.
            例如 [0, .10, .5, .90, 1.] 或 [.05, .5, .95]
            'quantiles' 和 'bins' 有且只能有一个不为 None
        bins :
            int or sequence[float]
            默认为 None
            在因子分组中使用的等宽 (按照因子值) 区间的数量
            或边界值序列, 允许不均匀的区间宽度.
            例如 [-4, -2, -0.5, 0, 10]
            'quantiles' 和 'bins' 有且只能有一个不为 None
        periods :
            int or sequence[int]
            远期收益的期数, 默认为 (1, 5, 10)
        max_loss :
            float
            默认为 0.25
            允许的丢弃因子数据的最大百分比 (0.00 到 1.00),
            计算比较输入因子索引中的项目数和输出 DataFrame 索引中的项目数.
            因子数据本身存在缺陷 (例如 NaN),
            没有提供足够的价格数据来计算所有因子值的远期收益，
            或者因为分组失败, 因此可以部分地丢弃因子数据
            设置 max_loss = 0 以停止异常捕获.
        zero_aware :
            bool
            默认为 False
            如果为True，则分别为正负因子值计算分位数。
            适用于您的信号聚集并且零是正值和负值的分界线的情况.


    所有属性列表
    ----------
        factor_data:返回因子值
            - 类型: pandas.Series
            - index: 为日期和股票代码的MultiIndex
        clean_factor_data: 去除 nan/inf, 整理后的因子值、forward_return 和分位数
            - 类型: pandas.DataFrame
            - index: 为日期和股票代码的MultiIndex
            - columns: 根据period选择后的forward_return
                    (如果调仓周期为1天, 那么 forward_return 为
                        [第二天的收盘价-今天的收盘价]/今天的收盘价),
                    因子值、行业分组、分位数数组、权重
        mean_return_by_quantile: 按分位数分组加权平均因子收益
            - 类型: pandas.DataFrame
            - index: 分位数分组
            - columns: 调仓周期
        mean_return_std_by_quantile: 按分位数分组加权因子收益标准差
            - 类型: pandas.DataFrame
            - index: 分位数分组
            - columns: 调仓周期
        mean_return_by_date: 按分位数及日期分组加权平均因子收益
            - 类型: pandas.DataFrame
            - index: 为日期和分位数的MultiIndex
            - columns: 调仓周期
        mean_return_std_by_date: 按分位数及日期分组加权因子收益标准差
            - 类型: pandas.DataFrame
            - index: 为日期和分位数的MultiIndex
            - columns: 调仓周期
        mean_return_by_group: 按分位数及行业分组加权平均因子收益
            - 类型: pandas.DataFrame
            - index: 为行业和分位数的MultiIndex
            - columns: 调仓周期
        mean_return_std_by_group: 按分位数及行业分组加权因子收益标准差
            - 类型: pandas.DataFrame
            - index: 为行业和分位数的MultiIndex
            - columns: 调仓周期
        mean_return_spread_by_quantile: 最高分位数因子收益减最低分位数因子收益每日均值
            - 类型: pandas.DataFrame
            - index: 日期
            - columns: 调仓周期
        mean_return_spread_std_by_quantile: 最高分位数因子收益减最低分位数因子收益每日标准差
            - 类型: pandas.DataFrame
            - index: 日期
            - columns: 调仓周期
        cumulative_return_by_quantile:各分位数每日累积收益
            - 类型: pandas.DataFrame
            - index: 日期
            - columns: 调仓周期和分位数
        cumulative_returns: 按因子值加权多空组合每日累积收益
            - 类型: pandas.DataFrame
            - index: 日期
            - columns: 调仓周期
        top_down_cumulative_returns: 做多最高分位做空最低分位多空组合每日累计收益
            - 类型: pandas.DataFrame
            - index: 日期
            - columns: 调仓周期
        ic: 信息比率
            - 类型: pandas.DataFrame
            - index: 日期
            - columns: 调仓周期
        ic_by_group: 分行业信息比率
            - 类型: pandas.DataFrame
            - index: 行业
            - columns: 调仓周期
        ic_monthly: 月度信息比率
            - 类型: pandas.DataFrame
            - index: 月度
            - columns: 调仓周期表
        quantile_turnover: 换手率
            - 类型: dict
            - 键: 调仓周期
                - index: 日期
                - columns: 分位数分组

    所有方法列表
    ----------
        calc_mean_return_by_quantile:
            计算按分位数分组加权因子收益和标准差
        calc_factor_returns:
            计算按因子值加权多空组合每日收益
        compute_mean_returns_spread:
            计算两个分位数相减的因子收益和标准差
        calc_factor_alpha_beta:
            计算因子的 alpha 和 beta
        calc_factor_information_coefficient:
            计算每日因子信息比率 (IC值)
        calc_mean_information_coefficient:
            计算因子信息比率均值 (IC值均值)
        calc_average_cumulative_return_by_quantile:
            按照当天的分位数算分位数未来和过去的收益均值和标准差
        calc_cumulative_return_by_quantile:
            计算各分位数每日累积收益
        calc_cumulative_returns:
            计算按因子值加权多空组合每日累积收益
        calc_top_down_cumulative_returns:
            计算做多最高分位做空最低分位多空组合每日累计收益
        calc_autocorrelation:
            根据调仓周期确定滞后期的每天计算因子自相关性
        calc_autocorrelation_n_days_lag:
            后 1 - n 天因子值自相关性
        calc_quantile_turnover_mean_n_days_lag:
            各分位数 1 - n 天平均换手率
        calc_ic_mean_n_days_lag:
            滞后 0 - n 天 forward return 信息比率

        plot_returns_table: 打印因子收益表
        plot_turnover_table: 打印换手率表
        plot_information_table: 打印信息比率(IC)相关表
        plot_quantile_statistics_table: 打印各分位数统计表
        plot_ic_ts: 画信息比率(IC)时间序列图
        plot_ic_hist: 画信息比率分布直方图
        plot_ic_qq: 画信息比率 qq 图
        plot_quantile_returns_bar: 画各分位数平均收益图
        plot_quantile_returns_violin: 画各分位数收益分布图
        plot_mean_quantile_returns_spread_time_series: 画最高分位减最低分位收益图
        plot_ic_by_group: 画按行业分组信息比率(IC)图
        plot_factor_auto_correlation: 画因子自相关图
        plot_top_bottom_quantile_turnover: 画最高最低分位换手率图
        plot_monthly_ic_heatmap: 画月度信息比率(IC)图
        plot_cumulative_returns: 画按因子值加权组合每日累积收益图
        plot_top_down_cumulative_returns: 画做多最大分位数做空最小分位数组合每日累积收益图
        plot_cumulative_returns_by_quantile: 画各分位数每日累积收益图
        plot_quantile_average_cumulative_return: 因子预测能力平均累计收益图
        plot_events_distribution: 画有效因子数量统计图

        create_summary_tear_sheet: 因子值特征分析
        create_returns_tear_sheet: 因子收益分析
        create_information_tear_sheet: 因子 IC 分析
        create_turnover_tear_sheet: 因子换手率分析
        create_event_returns_tear_sheet: 因子预测能力分析
        create_full_tear_sheet: 全部分析

        plot_disable_chinese_label: 关闭中文图例显示
        """

    def __init__(self, factor, prices, groupby=None, weights=1.0,
                 quantiles=None, bins=None, periods=(1, 5, 10),
                 binning_by_group=False, max_loss=0.25, zero_aware=False):

        self.factor = factor
        self.prices = prices
        self.groupby = groupby
        self.weights = weights

        self._quantiles = quantiles
        self._bins = bins
        self._periods = ensure_tuple(periods)
        self._binning_by_group = binning_by_group
        self._max_loss = max_loss
        self._zero_aware = zero_aware

        self.__gen_clean_factor_and_forward_returns()

    def __gen_clean_factor_and_forward_returns(self):
        """格式化因子数据和定价数据"""

        factor_data = self.factor
        if isinstance(factor_data, pd.DataFrame):
            factor_data = factor_data.stack(dropna=False)

        stocks = list(factor_data.index.get_level_values(1).drop_duplicates())
        start_date = min(factor_data.index.get_level_values(0))
        end_date = max(factor_data.index.get_level_values(0))

        if hasattr(self.prices, "__call__"):
            prices = self.prices(securities=stocks,
                                 start_date=start_date,
                                 end_date=end_date,
                                 period=max(self._periods))
            prices = prices.loc[~prices.index.duplicated()]
        else:
            prices = self.prices
        self._prices = prices

        if hasattr(self.groupby, "__call__"):
            groupby = self.groupby(securities=stocks,
                                   start_date=start_date,
                                   end_date=end_date)
        else:
            groupby = self.groupby
        self._groupby = groupby

        if hasattr(self.weights, "__call__"):
            weights = self.weights(stocks,
                                   start_date=start_date,
                                   end_date=end_date)
        else:
            weights = self.weights
        self._weights = weights

        self._clean_factor_data = get_clean_factor_and_forward_returns(
            factor_data,
            prices,
            groupby=groupby,
            weights=weights,
            binning_by_group=self._binning_by_group,
            quantiles=self._quantiles,
            bins=self._bins,
            periods=self._periods,
            max_loss=self._max_loss,
            zero_aware=self._zero_aware
        )

    @property
    def clean_factor_data(self):
        return self._clean_factor_data

    @property
    def _factor_quantile(self):
        data = self.clean_factor_data
        if not data.empty:
            return max(data.factor_quantile)
        else:
            _quantiles = self._quantiles
            _bins = self._bins
            _zero_aware = self._zero_aware
            get_len = lambda x: len(x) - 1 if isinstance(x, Iterable) else int(x)
            if _quantiles is not None and _bins is None and not _zero_aware:
                return get_len(_quantiles)
            elif _quantiles is not None and _bins is None and _zero_aware:
                return int(_quantiles) // 2 * 2
            elif _bins is not None and _quantiles is None and not _zero_aware:
                return get_len(_bins)
            elif _bins is not None and _quantiles is None and _zero_aware:
                return int(_bins) // 2 * 2

    @lru_cache(16)
    def calc_mean_return_by_quantile(self, by_date=False, by_group=False,
                                     demeaned=False, group_adjust=False):
        """计算按分位数分组因子收益和标准差

        因子收益为收益按照 weight 列中权重的加权平均值

        参数:
        by_date:
        - True: 按天计算收益
        - False: 不按天计算收益
        by_group:
        - True: 按行业计算收益
        - False: 不按行业计算收益
        demeaned:
        - True: 使用超额收益计算各分位数收益，超额收益=收益-基准收益
                (基准收益被认为是每日所有股票收益按照weight列中权重的加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性收益计算各分位数收益，行业中性收益=收益-行业收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        """
        return pef.mean_return_by_quantile(self._clean_factor_data,
                                           by_date=by_date,
                                           by_group=by_group,
                                           demeaned=demeaned,
                                           group_adjust=group_adjust)

    @lru_cache(4)
    def calc_factor_returns(self, demeaned=True, group_adjust=False):
        """计算按因子值加权组合每日收益

        权重 = 每日因子值 / 每日因子值的绝对值的和
        正的权重代表买入, 负的权重代表卖出

        参数:
        demeaned:
        - True: 对权重去均值 (每日权重 = 每日权重 - 每日权重的均值), 使组合转换为 cash-neutral 多空组合
        - False: 不对权重去均值
        group_adjust:
        - True: 对权重分行业去均值 (每日权重 = 每日权重 - 每日各行业权重的均值)，
                使组合转换为 industry-neutral 多空组合
        - False: 不对权重分行业去均值
        """
        return pef.factor_returns(self._clean_factor_data,
                                  demeaned=demeaned,
                                  group_adjust=group_adjust)

    def compute_mean_returns_spread(self, upper_quant=None, lower_quant=None,
                                    by_date=True, by_group=False,
                                    demeaned=False, group_adjust=False):
        """计算两个分位数相减的因子收益和标准差

        参数:
        upper_quant: 用 upper_quant 选择的分位数减去 lower_quant 选择的分位数
        lower_quant: 用 upper_quant 选择的分位数减去 lower_quant 选择的分位数
        by_date:
        - True: 按天计算两个分位数相减的因子收益和标准差
        - False: 不按天计算两个分位数相减的因子收益和标准差
        by_group:
        - True: 分行业计算两个分位数相减的因子收益和标准差
        - False: 不分行业计算两个分位数相减的因子收益和标准差
        demeaned:
        - True: 使用超额收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性收益
        """
        upper_quant = upper_quant if upper_quant is not None else self._factor_quantile
        lower_quant = lower_quant if lower_quant is not None else 1
        if ((not 1 <= upper_quant <= self._factor_quantile) or
            (not 1 <= lower_quant <= self._factor_quantile)):
            raise ValueError("upper_quant 和 low_quant 的取值范围为 1 - %s 的整数"
                             % self._factor_quantile)
        mean, std = self.calc_mean_return_by_quantile(by_date=by_date, by_group=by_group,
                                                      demeaned=demeaned, group_adjust=group_adjust,)
        mean = mean.apply(rate_of_return, axis=0)
        std = std.apply(rate_of_return, axis=0)
        return pef.compute_mean_returns_spread(mean_returns=mean,
                                               upper_quant=upper_quant,
                                               lower_quant=lower_quant,
                                               std_err=std)

    @lru_cache(4)
    def calc_factor_alpha_beta(self, demeaned=True, group_adjust=False):
        """计算因子的 alpha 和 beta

        因子值加权组合每日收益 = beta * 市场组合每日收益 + alpha

        因子值加权组合每日收益计算方法见 calc_factor_returns 函数
        市场组合每日收益是每日所有股票收益按照weight列中权重加权的均值
        结果中的 alpha 是年化 alpha

        参数:
        demeaned:
        详见 calc_factor_returns 中 demeaned 参数
        - True: 对因子值加权组合每日收益的权重去均值 (每日权重 = 每日权重 - 每日权重的均值),
                使组合转换为cash-neutral多空组合
        - False: 不对权重去均值
        group_adjust:
        详见 calc_factor_returns 中 group_adjust 参数
        - True: 对权重分行业去均值 (每日权重 = 每日权重 - 每日各行业权重的均值)，
                使组合转换为 industry-neutral 多空组合
        - False: 不对权重分行业去均值
        """
        return pef.factor_alpha_beta(self._clean_factor_data,
                                     demeaned=demeaned,
                                     group_adjust=group_adjust)

    @lru_cache(8)
    def calc_factor_information_coefficient(self, group_adjust=False, by_group=False, method=None):
        """计算每日因子信息比率 (IC值)

        参数:
        group_adjust:
        - True: 使用行业中性收益计算 IC (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性收益
        by_group:
        - True: 分行业计算 IC
        - False: 不分行业计算 IC
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用普通相关系数计算IC值
        """
        if method is None:
            method = 'rank'
        if method not in ('rank', 'normal'):
            raise ValueError("`method` should be chosen from ('rank' | 'normal')")

        if method == 'rank':
            method = spearmanr
        elif method == 'normal':
            method = pearsonr
        return pef.factor_information_coefficient(self._clean_factor_data,
                                                  group_adjust=group_adjust,
                                                  by_group=by_group,
                                                  method=method)

    @lru_cache(16)
    def calc_mean_information_coefficient(self, group_adjust=False, by_group=False,
                                          by_time=None, method=None):
        """计算因子信息比率均值 (IC值均值)

        参数:
        group_adjust:
        - True: 使用行业中性收益计算 IC (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性收益
        by_group:
        - True: 分行业计算 IC
        - False: 不分行业计算 IC
        by_time:
        - 'Y': 按年求均值
        - 'M': 按月求均值
        - None: 对所有日期求均值
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用普通相关系数计算IC值
        """
        if method is None:
            method = 'rank'
        if method not in ('rank', 'normal'):
            raise ValueError("`method` should be chosen from ('rank' | 'normal')")

        if method == 'rank':
            method = spearmanr
        elif method == 'normal':
            method = pearsonr
        return pef.mean_information_coefficient(
            self._clean_factor_data,
            group_adjust=group_adjust,
            by_group=by_group,
            by_time=by_time,
            method=method
        )

    @lru_cache(16)
    def calc_average_cumulative_return_by_quantile(self, periods_before, periods_after,
                                                   demeaned=False, group_adjust=False):
        """按照当天的分位数算分位数未来和过去的收益均值和标准差

        参数:
        periods_before: 计算过去的天数
        periods_after: 计算未来的天数
        demeaned:
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        return pef.average_cumulative_return_by_quantile(
            self._clean_factor_data,
            prices=self._prices,
            periods_before=periods_before,
            periods_after=periods_after,
            demeaned=demeaned,
            group_adjust=group_adjust
        )

    @lru_cache(2)
    def calc_autocorrelation(self, rank=True):
        """根据调仓周期确定滞后期的每天计算因子自相关性

        当日因子值和滞后period天的因子值的自相关性

        参数:
        rank:
        - True: 秩相关系数
        - False: 普通相关系数
        """
        return pd.concat(
            [
                pef.factor_autocorrelation(self._clean_factor_data,
                                           period, rank=rank)
                for period in self._periods
            ],
            axis=1,
            keys=list(map(convert_to_forward_returns_columns, self._periods))
        )

    @lru_cache(None)
    def calc_quantile_turnover_mean_n_days_lag(self, n=10):
        """各分位数滞后1天到n天的换手率均值

        参数:
        n: 滞后 1 天到 n 天的换手率
        """
        quantile_factor = self._clean_factor_data['factor_quantile']

        quantile_turnover_rate = pd.concat(
            [pd.Series([pef.quantile_turnover(quantile_factor, q, p).mean()
                        for q in range(1, int(quantile_factor.max()) + 1)],
                       index=range(1, int(quantile_factor.max()) + 1),
                       name=p)
             for p in range(1, n + 1)],
            axis=1, keys='lag_' + pd.Index(range(1, n + 1)).astype(str)
        ).T
        quantile_turnover_rate.columns.name = 'factor_quantile'

        return quantile_turnover_rate

    @lru_cache(None)
    def calc_autocorrelation_n_days_lag(self, n=10, rank=False):
        """滞后1-n天因子值自相关性

        参数:
        n: 滞后1天到n天的因子值自相关性
        rank:
        - True: 秩相关系数
        - False: 普通相关系数
        """
        return pd.Series(
            [
                pef.factor_autocorrelation(self._clean_factor_data, p, rank=rank).mean()
                for p in range(1, n + 1)
            ],
            index='lag_' + pd.Index(range(1, n + 1)).astype(str)
        )

    @lru_cache(None)
    def _calc_ic_mean_n_day_lag(self, n, group_adjust=False, by_group=False, method=None):
        if method is None:
            method = 'rank'
        if method not in ('rank', 'normal'):
            raise ValueError("`method` should be chosen from ('rank' | 'normal')")

        if method == 'rank':
            method = spearmanr
        elif method == 'normal':
            method = pearsonr

        factor_data = self._clean_factor_data.copy()
        factor_value = factor_data['factor'].unstack('asset')

        factor_data['factor'] = factor_value.shift(n).stack(dropna=True)
        if factor_data.dropna().empty:
            return pd.Series(np.nan, index=pef.get_forward_returns_columns(factor_data.columns))
        ac = pef.factor_information_coefficient(
            factor_data.dropna(),
            group_adjust=group_adjust, by_group=by_group,
            method=method
        )
        return ac.mean(level=('group' if by_group else None))

    def calc_ic_mean_n_days_lag(self, n=10, group_adjust=False, by_group=False, method=None):
        """滞后 0 - n 天因子收益信息比率(IC)的均值

        滞后 n 天 IC 表示使用当日因子值和滞后 n 天的因子收益计算 IC

        参数:
        n: 滞后0-n天因子收益的信息比率(IC)的均值
        group_adjust:
        - True: 使用行业中性收益计算 IC (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性收益
        by_group:
        - True: 分行业计算 IC
        - False: 不分行业计算 IC
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用普通相关系数计算IC值
        """
        ic_mean = [self.calc_factor_information_coefficient(
            group_adjust=group_adjust, by_group=by_group, method=method,
        ).mean(level=('group' if by_group else None))]

        for lag in range(1, n + 1):
            ic_mean.append(self._calc_ic_mean_n_day_lag(
                n=lag,
                group_adjust=group_adjust,
                by_group=by_group,
                method=method
            ))
        if not by_group:
            ic_mean = pd.concat(ic_mean, keys='lag_' + pd.Index(range(n + 1)).astype(str), axis=1)
            ic_mean = ic_mean.T
        else:
            ic_mean = pd.concat(ic_mean, keys='lag_' + pd.Index(range(n + 1)).astype(str), axis=0)
        return ic_mean

    @property
    def mean_return_by_quantile(self):
        """收益分析

        用来画分位数收益的柱状图

        返回 pandas.DataFrame, index 是 factor_quantile, 值是(1, 2, 3, 4, 5),
        column 是 period 的值 (1, 5, 10)
        """
        mean_ret_quantile, _ = self.calc_mean_return_by_quantile(
            by_date=False,
            by_group=False,
            demeaned=False,
            group_adjust=False,
        )
        mean_compret_quantile = mean_ret_quantile.apply(rate_of_return, axis=0)
        return mean_compret_quantile

    @property
    def mean_return_std_by_quantile(self):
        """收益分析

        用来画分位数收益的柱状图

        返回 pandas.DataFrame, index 是 factor_quantile, 值是(1, 2, 3, 4, 5),
        column 是 period 的值 (1, 5, 10)
        """
        _, mean_ret_std_quantile = self.calc_mean_return_by_quantile(
            by_date=False,
            by_group=False,
            demeaned=False,
            group_adjust=False,
        )
        mean_ret_std_quantile = mean_ret_std_quantile.apply(rate_of_return, axis=0)
        return mean_ret_std_quantile

    @property
    def _mean_return_by_date(self):
        _mean_return_by_date, _ = self.calc_mean_return_by_quantile(
            by_date=True,
            by_group=False,
            demeaned=False,
            group_adjust=False,
        )
        return _mean_return_by_date

    @property
    def mean_return_by_date(self):
        mean_return_by_date = self._mean_return_by_date.apply(rate_of_return, axis=0)
        return mean_return_by_date

    @property
    def mean_return_std_by_date(self):
        _, std_quant_daily = self.calc_mean_return_by_quantile(
            by_date=True,
            demeaned=False,
            by_group=False,
            group_adjust=False,
        )
        mean_return_std_by_date = std_quant_daily.apply(std_conversion, axis=0)

        return mean_return_std_by_date

    @property
    def mean_return_by_group(self):
        """分行业的分位数收益

        返回值:
            MultiIndex 的 DataFrame
            index 分别是分位数、 行业名称,  column 是 period  (1, 5, 10)
        """
        mean_return_group, _ = self.calc_mean_return_by_quantile(
            by_date=False,
            by_group=True,
            demeaned=True,
            group_adjust=False,
        )
        mean_return_group = mean_return_group.apply(rate_of_return, axis=0)
        return mean_return_group

    @property
    def mean_return_std_by_group(self):
        _, mean_return_std_group = self.calc_mean_return_by_quantile(
            by_date=False,
            by_group=True,
            demeaned=True,
            group_adjust=False,
        )
        mean_return_std_group = mean_return_std_group.apply(rate_of_return, axis=0)
        return mean_return_std_group

    @property
    def mean_return_spread_by_quantile(self):
        mean_return_spread_by_quantile, _ = self.compute_mean_returns_spread()
        return mean_return_spread_by_quantile

    @property
    def mean_return_spread_std_by_quantile(self):
        _, std_spread_quant = self.compute_mean_returns_spread()
        return std_spread_quant

    @lru_cache(5)
    def calc_cumulative_return_by_quantile(self, period=None, demeaned=False, group_adjust=False):
        """计算指定调仓周期的各分位数每日累积收益

        参数:
        period: 指定调仓周期
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        if period is None:
            period = self._periods[0]
        period_col = convert_to_forward_returns_columns(period)

        factor_returns = self.calc_mean_return_by_quantile(
            by_date=True, demeaned=demeaned, group_adjust=group_adjust
        )[0][period_col].unstack('factor_quantile')

        cum_ret = factor_returns.apply(pef.cumulative_returns, period=period)

        return cum_ret

    @lru_cache(20)
    def calc_cumulative_returns(self, period=None,
                                demeaned=False, group_adjust=False):
        """计算指定调仓周期的按因子值加权组合每日累积收益

        当 period > 1 时，组合的累积收益计算方法为：
        组合每日收益 = （从第0天开始每period天一调仓的组合每日收益 +
                        从第1天开始每period天一调仓的组合每日收益 + ... +
                        从第period-1天开始每period天一调仓的组合每日收益) / period
        组合累积收益 = 组合每日收益的累积

        参数:
        period: 指定调仓周期
        demeaned:
        详见 calc_factor_returns 中 demeaned 参数
        - True: 对权重去均值 (每日权重 = 每日权重 - 每日权重的均值), 使组合转换为 cash-neutral 多空组合
        - False: 不对权重去均值
        group_adjust:
        详见 calc_factor_returns 中 group_adjust 参数
        - True: 对权重分行业去均值 (每日权重 = 每日权重 - 每日各行业权重的均值)，
                使组合转换为 industry-neutral 多空组合
        - False: 不对权重分行业去均值
        """
        if period is None:
            period = self._periods[0]
        period_col = convert_to_forward_returns_columns(period)
        factor_returns = self.calc_factor_returns(
            demeaned=demeaned, group_adjust=group_adjust
        )[period_col]

        return pef.cumulative_returns(factor_returns, period=period)

    @lru_cache(20)
    def calc_top_down_cumulative_returns(self, period=None,
                                         demeaned=False, group_adjust=False):
        """计算做多最大分位，做空最小分位组合每日累积收益

        参数:
        period: 指定调仓周期
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        if period is None:
            period = self._periods[0]
        period_col = convert_to_forward_returns_columns(period)
        mean_returns, _ = self.calc_mean_return_by_quantile(
            by_date=True, by_group=False,
            demeaned=demeaned, group_adjust=group_adjust,
        )
        mean_returns = mean_returns.apply(rate_of_return, axis=0)

        upper_quant = mean_returns[period_col].xs(self._factor_quantile,
                                                  level='factor_quantile')
        lower_quant = mean_returns[period_col].xs(1,
                                                  level='factor_quantile')
        return pef.cumulative_returns(upper_quant - lower_quant, period=period)

    @property
    def ic(self):
        """IC 分析, 日度 ic

        返回 DataFrame, index 是时间,  column 是 period 的值 (1, 5, 10)
        """
        return self.calc_factor_information_coefficient()

    @property
    def ic_by_group(self):
        """行业 ic"""
        return self.calc_mean_information_coefficient(by_group=True)

    @property
    def ic_monthly(self):
        ic_monthly = self.calc_mean_information_coefficient(group_adjust=False,
                                                            by_group=False,
                                                            by_time="M").copy()
        ic_monthly.index = ic_monthly.index.map(lambda x: x.strftime('%Y-%m'))
        return ic_monthly

    @cached_property
    def quantile_turnover(self):
        """换手率分析

        返回值一个 dict, key 是 period, value 是一个 DataFrame(index 是日期, column 是分位数)
        """

        quantile_factor = self._clean_factor_data['factor_quantile']

        quantile_turnover_rate = {
            convert_to_forward_returns_columns(p):
            pd.concat([pef.quantile_turnover(quantile_factor, q, p)
                       for q in range(1, int(quantile_factor.max()) + 1)],
                      axis=1)
            for p in self._periods
        }

        return quantile_turnover_rate

    @property
    def cumulative_return_by_quantile(self):
        return {
            convert_to_forward_returns_columns(p):
            self.calc_cumulative_return_by_quantile(period=p)
            for p in self._periods
        }

    @property
    def cumulative_returns(self):
        return pd.concat([self.calc_cumulative_returns(period=period)
                          for period in self._periods],
                         axis=1,
                         keys=list(map(convert_to_forward_returns_columns,
                                       self._periods)))

    @property
    def top_down_cumulative_returns(self):
        return pd.concat([self.calc_top_down_cumulative_returns(period=period)
                          for period in self._periods],
                         axis=1,
                         keys=list(map(convert_to_forward_returns_columns,
                                       self._periods)))

    def plot_returns_table(self, demeaned=False, group_adjust=False):
        """打印因子收益表

        参数:
        demeaned:
        - True: 使用超额收益计算 (基准收益被认为是每日所有股票收益按照weight列中权重的加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        """
        mean_return_by_quantile = self.calc_mean_return_by_quantile(
            by_date=False, by_group=False,
            demeaned=demeaned, group_adjust=group_adjust,
        )[0].apply(rate_of_return, axis=0)

        mean_returns_spread, _ = self.compute_mean_returns_spread(
            upper_quant=self._factor_quantile,
            lower_quant=1,
            by_date=True,
            by_group=False,
            demeaned=demeaned,
            group_adjust=group_adjust,
        )

        pl.plot_returns_table(
            self.calc_factor_alpha_beta(demeaned=demeaned),
            mean_return_by_quantile,
            mean_returns_spread
        )

    def plot_turnover_table(self):
        """打印换手率表"""
        pl.plot_turnover_table(
            self.calc_autocorrelation(),
            self.quantile_turnover
        )

    def plot_information_table(self, group_adjust=False, method=None):
        """打印信息比率 (IC)相关表

        参数:
        group_adjust:
        - True：使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False：不使用行业中性收益
        method：
        - 'rank'：用秩相关系数计算IC值
        - 'normal':用相关系数计算IC值
        """
        ic = self.calc_factor_information_coefficient(
            group_adjust=group_adjust,
            by_group=False,
            method=method
        )
        pl.plot_information_table(ic)

    def plot_quantile_statistics_table(self):
        """打印各分位数统计表"""
        pl.plot_quantile_statistics_table(self._clean_factor_data)

    def plot_ic_ts(self, group_adjust=False, method=None):
        """画信息比率(IC)时间序列图

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal':用相关系数计算IC值
        """
        ic = self.calc_factor_information_coefficient(
            group_adjust=group_adjust, by_group=False, method=method
        )
        pl.plot_ic_ts(ic)

    def plot_ic_hist(self, group_adjust=False, method=None):
        """画信息比率分布直方图

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用相关系数计算IC值
        """
        ic = self.calc_factor_information_coefficient(
            group_adjust=group_adjust,
            by_group=False,
            method=method
        )
        pl.plot_ic_hist(ic)

    def plot_ic_qq(self, group_adjust=False, method=None, theoretical_dist=None):
        """画信息比率 qq 图

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用相关系数计算IC值
        theoretical_dist:
        - 'norm': 正态分布
        - 't': t 分布
        """
        theoretical_dist = 'norm' if theoretical_dist is None else theoretical_dist
        theoretical_dist = morestats._parse_dist_kw(theoretical_dist)
        ic = self.calc_factor_information_coefficient(
            group_adjust=group_adjust,
            by_group=False,
            method=method,
        )
        pl.plot_ic_qq(ic, theoretical_dist=theoretical_dist)

    def plot_quantile_returns_bar(self, by_group=False,
                                  demeaned=False, group_adjust=False):
        """画各分位数平均收益图

        参数:
        by_group:
        - True: 各行业的各分位数平均收益图
        - False: 各分位数平均收益图
        demeaned:
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        mean_return_by_quantile = self.calc_mean_return_by_quantile(
            by_date=False, by_group=by_group,
            demeaned=demeaned, group_adjust=group_adjust,
        )[0].apply(rate_of_return, axis=0)

        pl.plot_quantile_returns_bar(
            mean_return_by_quantile, by_group=by_group, ylim_percentiles=None
        )

    def plot_quantile_returns_violin(self, demeaned=False, group_adjust=False,
                                     ylim_percentiles=(1, 99)):
        """画各分位数收益分布图

        参数:
        demeaned:
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        plot_quantile_returns_violin: 有效收益分位数(单位为百分之). 画图时y轴的范围为有效收益的最大/最小值.
                                      例如 (1, 99) 代表收益的从小到大排列的 1% 分位到 99% 分位为有效收益.
        """
        mean_return_by_date = self.calc_mean_return_by_quantile(
            by_date=True, by_group=False,
            demeaned=demeaned, group_adjust=group_adjust
        )[0].apply(rate_of_return, axis=0)

        pl.plot_quantile_returns_violin(mean_return_by_date,
                                        ylim_percentiles=ylim_percentiles)

    def plot_mean_quantile_returns_spread_time_series(
        self, demeaned=False, group_adjust=False, bandwidth=1
    ):
        """画最高分位减最低分位收益图

        参数:
        demeaned:
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        bandwidth: n, 加减 n 倍当日标准差
        """
        mean_returns_spread, mean_returns_spread_std = self.compute_mean_returns_spread(
            upper_quant=self._factor_quantile,
            lower_quant=1,
            by_date=True,
            by_group=False,
            demeaned=demeaned,
            group_adjust=group_adjust,
        )

        pl.plot_mean_quantile_returns_spread_time_series(
            mean_returns_spread, std_err=mean_returns_spread_std,
            bandwidth=bandwidth
        )

    def plot_ic_by_group(self, group_adjust=False, method=None):
        """画按行业分组信息比率(IC)图

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        method:
        - 'rank': 用秩相关系数计算IC值
        - 'normal': 用相关系数计算IC值
        """
        ic_by_group = self.calc_mean_information_coefficient(
            group_adjust=group_adjust,
            by_group=True,
            method=method
        )
        pl.plot_ic_by_group(ic_by_group)

    def plot_factor_auto_correlation(self, periods=None, rank=True):
        """画因子自相关图

        参数:
        periods: 滞后周期
        rank:
        - True: 用秩相关系数
        - False: 用相关系数
        """
        if periods is None:
            periods = self._periods
        if not isinstance(periods, Iterable):
            periods = (periods,)
        periods = tuple(periods)
        for p in periods:
            if p in self._periods:
                pl.plot_factor_rank_auto_correlation(
                    self.calc_autocorrelation(rank=rank)[
                        convert_to_forward_returns_columns(p)
                    ],
                    period=p
                )

    def plot_top_bottom_quantile_turnover(self, periods=None):
        """画最高最低分位换手率图

        参数:
        periods: 调仓周期
        """
        quantile_turnover = self.quantile_turnover

        if periods is None:
            periods = self._periods
        if not isinstance(periods, Iterable):
            periods = (periods,)
        periods = tuple(periods)
        for p in periods:
            if p in self._periods:
                pl.plot_top_bottom_quantile_turnover(
                    quantile_turnover[convert_to_forward_returns_columns(p)],
                    period=p
                )

    def plot_monthly_ic_heatmap(self, group_adjust=False):
        """画月度信息比率(IC)图

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        """
        ic_monthly = self.calc_mean_information_coefficient(
            group_adjust=group_adjust, by_group=False, by_time="M"
        )
        pl.plot_monthly_ic_heatmap(ic_monthly)

    def plot_cumulative_returns(self, period=None, demeaned=False,
                                group_adjust=False):
        """画按因子值加权组合每日累积收益图

        参数:
        periods: 调仓周期
        demeaned:
        详见 calc_factor_returns 中 demeaned 参数
        - True: 对因子值加权组合每日收益的权重去均值 (每日权重 = 每日权重 - 每日权重的均值),
                使组合转换为cash-neutral多空组合
        - False: 不对权重去均值
        group_adjust:
        详见 calc_factor_returns 中 group_adjust 参数
        - True: 对权重分行业去均值 (每日权重 = 每日权重 - 每日各行业权重的均值)，
                使组合转换为 industry-neutral 多空组合
        - False: 不对权重分行业去均值
        """
        if period is None:
            period = self._periods
        if not isinstance(period, Iterable):
            period = (period,)
        period = tuple(period)
        factor_returns = self.calc_factor_returns(demeaned=demeaned,
                                                  group_adjust=group_adjust)
        for p in period:
            if p in self._periods:
                pl.plot_cumulative_returns(
                    factor_returns[convert_to_forward_returns_columns(p)],
                    period=p
                )

    def plot_top_down_cumulative_returns(self, period=None, demeaned=False, group_adjust=False):
        """画做多最大分位数做空最小分位数组合每日累积收益图

        period: 指定调仓周期
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        if period is None:
            period = self._periods
        if not isinstance(period, Iterable):
            period = (period, )
        period = tuple(period)
        for p in period:
            if p in self._periods:
                factor_return = self.calc_top_down_cumulative_returns(
                    period=p, demeaned=demeaned, group_adjust=group_adjust,
                )
                pl.plot_top_down_cumulative_returns(
                    factor_return, period=p
                )

    def plot_cumulative_returns_by_quantile(self, period=None, demeaned=False,
                                            group_adjust=False):
        """画各分位数每日累积收益图

        参数:
        period: 调仓周期
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        if period is None:
            period = self._periods
        if not isinstance(period, Iterable):
            period = (period,)
        period = tuple(period)
        mean_return_by_date, _ = self.calc_mean_return_by_quantile(
            by_date=True, by_group=False, demeaned=demeaned, group_adjust=group_adjust,
        )
        for p in period:
            if p in self._periods:
                pl.plot_cumulative_returns_by_quantile(
                    mean_return_by_date[convert_to_forward_returns_columns(p)],
                    period=p
                )

    def plot_quantile_average_cumulative_return(self, periods_before=5, periods_after=10,
                                                by_quantile=False, std_bar=False,
                                                demeaned=False, group_adjust=False):
        """因子预测能力平均累计收益图

        参数:
        periods_before: 计算过去的天数
        periods_after: 计算未来的天数
        by_quantile: 是否各分位数分别显示因子预测能力平均累计收益图
        std_bar:
        - True: 显示标准差
        - False: 不显示标准差
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        """
        average_cumulative_return_by_q = self.calc_average_cumulative_return_by_quantile(
            periods_before=periods_before, periods_after=periods_after,
            demeaned=demeaned, group_adjust=group_adjust
        )
        pl.plot_quantile_average_cumulative_return(average_cumulative_return_by_q,
                                                   by_quantile=by_quantile,
                                                   std_bar=std_bar,
                                                   periods_before=periods_before,
                                                   periods_after=periods_after)

    def plot_events_distribution(self, num_days=5):
        """画有效因子数量统计图

        参数:
        num_days: 统计间隔天数
        """
        pl.plot_events_distribution(
            events=self._clean_factor_data['factor'],
            num_days=num_days,
            full_dates=pd.to_datetime(self.factor.index.get_level_values('date').unique())
        )

    def create_summary_tear_sheet(self, demeaned=False, group_adjust=False):
        """因子值特征分析

        参数:
        demeaned:
        - True: 对每日因子收益去均值求得因子收益表
        - False: 因子收益表
        group_adjust:
        - True: 按行业对因子收益去均值后求得因子收益表
        - False: 因子收益表
        """
        self.plot_quantile_statistics_table()
        self.plot_returns_table(demeaned=demeaned, group_adjust=group_adjust)
        self.plot_quantile_returns_bar(by_group=False, demeaned=demeaned, group_adjust=group_adjust)
        pl.plt.show()
        self.plot_information_table(group_adjust=group_adjust)
        self.plot_turnover_table()

    def create_returns_tear_sheet(self, demeaned=False, group_adjust=False, by_group=False):
        """因子值特征分析

        参数:
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算 (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        by_group:
        - True: 画各行业的各分位数平均收益图
        - False: 不画各行业的各分位数平均收益图
        """
        self.plot_returns_table(demeaned=demeaned, group_adjust=group_adjust)
        self.plot_quantile_returns_bar(by_group=False,
                                       demeaned=demeaned,
                                       group_adjust=group_adjust)
        pl.plt.show()
        self.plot_cumulative_returns(
            period=None, demeaned=demeaned, group_adjust=group_adjust
        )
        pl.plt.show()
        self.plot_cumulative_returns_by_quantile(period=None,
                                                 demeaned=demeaned,
                                                 group_adjust=group_adjust)
        self.plot_top_down_cumulative_returns(period=None,
                                              demeaned=demeaned,
                                              group_adjust=group_adjust)
        pl.plt.show()
        self.plot_mean_quantile_returns_spread_time_series(
            demeaned=demeaned, group_adjust=group_adjust
        )
        pl.plt.show()
        if by_group:
            self.plot_quantile_returns_bar(by_group=True,
                                           demeaned=demeaned,
                                           group_adjust=group_adjust)
            pl.plt.show()

        self.plot_quantile_returns_violin(demeaned=demeaned,
                                          group_adjust=group_adjust)
        pl.plt.show()

    def create_information_tear_sheet(self, group_adjust=False, by_group=False):
        """因子 IC 分析

        参数:
        group_adjust:
        - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
        - False: 不使用行业中性收益
        by_group:
        - True: 画按行业分组信息比率(IC)图
        - False: 画月度信息比率(IC)图
        """
        self.plot_ic_ts(group_adjust=group_adjust, method=None)
        pl.plt.show()
        self.plot_ic_qq(group_adjust=group_adjust)
        pl.plt.show()
        if by_group:
            self.plot_ic_by_group(group_adjust=group_adjust, method=None)
        else:
            self.plot_monthly_ic_heatmap(group_adjust=group_adjust)
        pl.plt.show()

    def create_turnover_tear_sheet(self, turnover_periods=None):
        """因子换手率分析

        参数:
        turnover_periods: 调仓周期
        """
        self.plot_turnover_table()
        self.plot_top_bottom_quantile_turnover(periods=turnover_periods)
        pl.plt.show()
        self.plot_factor_auto_correlation(periods=turnover_periods)
        pl.plt.show()

    def create_event_returns_tear_sheet(self, avgretplot=(5, 15),
                                        demeaned=False, group_adjust=False,
                                        std_bar=False):
        """因子预测能力分析

        参数:
        avgretplot: tuple 因子预测的天数
        -(计算过去的天数, 计算未来的天数)
        demeaned:
        详见 calc_mean_return_by_quantile 中 demeaned 参数
        - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False: 不使用超额收益
        group_adjust:
        详见 calc_mean_return_by_quantile 中 group_adjust 参数
        - True: 使用行业中性化后的收益计算累积收益
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False: 不使用行业中性化后的收益
        std_bar:
        - True: 显示标准差
        - False: 不显示标准差
        """
        before, after = avgretplot
        self.plot_quantile_average_cumulative_return(
            periods_before=before, periods_after=after,
            by_quantile=False, std_bar=False,
            demeaned=demeaned, group_adjust=group_adjust
        )
        pl.plt.show()
        if std_bar:
            self.plot_quantile_average_cumulative_return(
                periods_before=before, periods_after=after,
                by_quantile=True, std_bar=True,
                demeaned=demeaned, group_adjust=group_adjust
            )
            pl.plt.show()

    def create_full_tear_sheet(self, demeaned=False, group_adjust=False, by_group=False,
                               turnover_periods=None, avgretplot=(5, 15), std_bar=False):
        """全部分析

        参数:
        demeaned:
        - True：使用超额收益计算 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
        - False：不使用超额收益
        group_adjust:
        - True：使用行业中性化后的收益计算
                (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
        - False：不使用行业中性化后的收益
        by_group:
        - True: 按行业展示
        - False: 不按行业展示
        turnover_periods: 调仓周期
        avgretplot: tuple 因子预测的天数
        -(计算过去的天数, 计算未来的天数)
        std_bar:
        - True: 显示标准差
        - False: 不显示标准差
        """
        self.plot_quantile_statistics_table()
        print("\n-------------------------\n")
        self.plot_returns_table(demeaned=demeaned, group_adjust=group_adjust)
        self.plot_quantile_returns_bar(by_group=False,
                                       demeaned=demeaned,
                                       group_adjust=group_adjust)
        pl.plt.show()
        self.plot_cumulative_returns(period=None, demeaned=demeaned, group_adjust=group_adjust)
        pl.plt.show()
        self.plot_top_down_cumulative_returns(period=None,
                                              demeaned=demeaned,
                                              group_adjust=group_adjust)
        pl.plt.show()
        self.plot_cumulative_returns_by_quantile(period=None,
                                                 demeaned=demeaned,
                                                 group_adjust=group_adjust)
        self.plot_mean_quantile_returns_spread_time_series(demeaned=demeaned,
                                                           group_adjust=group_adjust)
        pl.plt.show()
        if by_group:
            self.plot_quantile_returns_bar(by_group=True,
                                           demeaned=demeaned,
                                           group_adjust=group_adjust)
            pl.plt.show()
        self.plot_quantile_returns_violin(demeaned=demeaned,
                                          group_adjust=group_adjust)
        pl.plt.show()
        print("\n-------------------------\n")
        self.plot_information_table(group_adjust=group_adjust)
        self.plot_ic_ts(group_adjust=group_adjust, method=None)
        pl.plt.show()
        self.plot_ic_qq(group_adjust=group_adjust)
        pl.plt.show()
        if by_group:
            self.plot_ic_by_group(group_adjust=group_adjust, method=None)
        else:
            self.plot_monthly_ic_heatmap(group_adjust=group_adjust)
        pl.plt.show()
        print("\n-------------------------\n")
        self.plot_turnover_table()
        self.plot_top_bottom_quantile_turnover(periods=turnover_periods)
        pl.plt.show()
        self.plot_factor_auto_correlation(periods=turnover_periods)
        pl.plt.show()
        print("\n-------------------------\n")
        before, after = avgretplot
        self.plot_quantile_average_cumulative_return(
            periods_before=before, periods_after=after,
            by_quantile=False, std_bar=False,
            demeaned=demeaned, group_adjust=group_adjust
        )
        pl.plt.show()
        if std_bar:
            self.plot_quantile_average_cumulative_return(
                periods_before=before, periods_after=after,
                by_quantile=True, std_bar=True,
                demeaned=demeaned, group_adjust=group_adjust
            )
            pl.plt.show()

    def plot_disable_chinese_label(self):
        """关闭中文图例显示

        画图时默认会从系统中查找中文字体显示以中文图例
        如果找不到中文字体则默认使用英文图例
        当找到中文字体但中文显示乱码时, 可调用此 API 关闭中文图例显示而使用英文
        """
        _use_chinese(False)

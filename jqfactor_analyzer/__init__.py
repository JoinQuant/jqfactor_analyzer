# -*- coding: utf-8 -*-

from .version import __version__
from .analyze import FactorAnalyzer
from .attribution import AttributionAnalysis
from .data import DataApi
from .preprocess import winsorize, winsorize_med, standardlize, neutralize
from .factor_cache import save_factor_values_by_group, get_factor_values_by_cache, get_cache_dir


def analyze_factor(
    factor, industry='jq_l1', quantiles=5, periods=(1, 5, 10),
    weight_method='avg', max_loss=0.25, allow_cache=True, show_data_progress=True
):
    """单因子分析

    输入:
        factor: pandas.DataFrame: 因子值, columns 为股票代码 (如 '000001.XSHE'),
                                          index 为 日期的 DatetimeIndex
                或 pandas.Series: 因子值, index 为日期和股票代码的 MultiIndex
        industry: 行业分类, 默认为 'jq_l1'
            - 'jq_l1': 聚宽一级行业
            - 'jq_l2': 聚宽二级行业
            - 'sw_l1': 申万一级行业
            - 'sw_l2': 申万二级行业
            - 'sw_l3': 申万三级行业
            - 'zjw': 证监会行业
        quantiles: 分位数数量, 默认为 5
        periods: 调仓周期, int 或 int 的 列表, 默认为 [1, 5, 10]
        weight_method: 计算分位数收益时的加权方法, 默认为 'avg'
            - 'avg': 等权重
            - 'mktcap': 按总市值加权
            - 'ln_mktcap': 按总市值的对数加权
            - 'cmktcap': 按流通市值加权
            - 'ln_cmktcap': 按流通市值的对数加权
        max_loss: 因重复值或nan值太多而无效的因子值的最大占比, 默认为 0.25
        allow_cache: 是否允许对价格,市值等信息进行本地缓存(按天缓存,初次运行可能比较慢,但后续重新获取对应区间的数据将非常快,且分析时仅消耗较小的jqdatasdk流量)
        show_data_progress: 是否展示数据获取的进度信息

    """

    dataapi = DataApi(industry=industry, weight_method=weight_method,
                      allow_cache=allow_cache, show_progress=show_data_progress)
    return FactorAnalyzer(factor,
                          quantiles=quantiles,
                          periods=periods,
                          max_loss=max_loss,
                          **dataapi.apis)


def attribution_analysis(
    weights, daily_return, style_type='style_pro', industry='sw_l1',
    use_cn=True, show_data_progress=True
):
    """归因分析

    用户需要提供的数据:
    1. 日度股票持仓权重 (加总不为 1 的剩余部分视为现金)
    2. 组合的的日度收益率 (使用 T 日持仓盘后的因子暴露与 T+1 日的收益进行归因分析)

    组合风格因子暴露 (含行业, country) = sum(组合权重 * 个股因子值), country 暴露为总的股票持仓权重
    组合风格收益率 (含行业, country) = sum(组合风格因子暴露 * factor_return)
    组合特异收益率 = 组合总收益率 - 组合风格收益率(含行业, country 或 cash)
    """
    return AttributionAnalysis(weights,
                               daily_return=daily_return,
                               style_type=style_type,
                               industry=industry,
                               use_cn=use_cn,
                               show_data_progress=show_data_progress)

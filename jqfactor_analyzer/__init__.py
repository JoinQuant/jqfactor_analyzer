# -*- coding: utf-8 -*-

from .version import __version__
from .analyze import FactorAnalyzer
from .data import DataApi


def analyze_factor(
    factor, industry='jq_l1', quantiles=5, periods=(1, 5, 10),
    weight_method='avg', max_loss=0.25 , allow_cache=True, show_data_progress=True
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

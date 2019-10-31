# -*- coding: utf-8 -*-


from functools import partial

import numpy as np
import pandas as pd
from fastcache import lru_cache

from .when import date2str


class DataApi(object):

    def __init__(self, price='close', fq='post',
                 industry='jq_l1', weight_method='avg'):
        """数据接口, 用于因子分析获取数据

        参数
        ----------
        price : 使用开盘价/收盘价计算收益 (请注意避免未来函数), 默认为 'close'
            - 'close': 使用当日收盘价和次日收盘价计算当日因子的远期收益
            - 'open' : 使用当日开盘价和次日开盘价计算当日因子的远期收益
        fq : 价格数据的复权方式, 默认为 'post'
            - 'post': 后复权
            - 'pre': 前复权
            - None: 不复权
        industry : 行业分类, 默认为 'jq_l1'
            - 'jq_l1': 聚宽一级行业
            - 'jq_l2': 聚宽二级行业
            - 'sw_l1': 申万一级行业
            - 'sw_l2': 申万二级行业
            - 'sw_l3': 申万三级行业
            - 'zjw': 证监会行业
        weight_method : 计算各分位收益时, 每只股票权重, 默认为 'avg'
            - 'avg': 等权重
            - 'mktcap': 按总市值加权
            - 'ln_mktcap': 按总市值的对数加权
            - 'cmktcap': 按流通市值加权
            - 'ln_cmktcap': 按流通市值的对数加权


        使用示例
        ----------
        from jqfactor_analyzer import DataApi, FactorAnalyzer

        api = DataApi(fq='pre', industry='sw_l1', weight_method='ln_mktcap')
        api.auth('username', 'password')

        factor = FactorAnalyzer(factor_data,
                                price=api.get_prices,
                                groupby=api.get_groupby,
                                weights=api.get_weights)
        # 或者
        # factor = FactorAnalyzer(factor_data, **api.apis)


        方法列表
        ----------
        auth : 登陆 jqdatasdk
          参数 :
            username : jqdatasdk 用户名
            username : jqdatasdk 密码
          返回值 :
            None

        get_prices : 价格数据获取接口
          参数 :
            securities : 股票代码列表
            start_date : 开始日期
            end_date : 结束日期
            count : 交易日长度
            (start_date 和 count)
          返回值 :
            pd.DataFrame
            价格数据, columns 为股票代码, index 为日期

        get_groupby : 行业分类数据获取接口
          参数 :
            securities : 股票代码列表
            start_date : 开始日期
            end_date : 结束日期
          返回值 :
            dict
            行业分类, {股票代码 -> 行业分类名称}

        get_weights : 股票权重获取接口
          参数 :
            securities : 股票代码列表
            start_date : 开始日期
            end_date : 结束日期
          返回值 :
            pd.DataFrame
            权重数据, columns 为股票代码, index 为日期


        属性列表
        ----------
        apis : dict, {'prices': get_prices, 'groupby': get_groupby,
                      'weights': get_weights}

        """
        try:
            import jqdata
            self._api = jqdata.apis
            self._api_name = 'jqdata'
        except ImportError:
            import jqdatasdk
            self._api = jqdatasdk
            self._api_name = 'jqdatasdk'

        valid_price = ('close', 'open')
        if price in valid_price:
            self.price = price
        else:
            ValueError("invalid 'price' parameter, "
                       "should be one of %s" % str(valid_price))

        valid_fq = ('post', 'pre', None)
        if fq in valid_fq:
            self.fq = fq
        else:
            raise ValueError("invalid 'fq' parameter, "
                             "should be one of %s" % str(valid_fq))

        valid_industry = ('sw_l1', 'sw_l2', 'sw_l3', 'jq_l1', 'jq_l2', 'zjw')
        if industry in valid_industry:
            self.industry = industry
        else:
            raise ValueError("invalid 'industry' parameter, "
                             "should be one of %s" % str(valid_industry))

        valid_weight_method = ('avg', 'mktcap', 'ln_mktcap',
                               'cmktcap', 'ln_cmktcap')
        if weight_method in valid_weight_method:
            self.weight_method = weight_method
        else:
            raise ValueError("invalid 'weight_method' parameter, "
                             "should be one of %s" % str(valid_weight_method))

    def auth(self, username='', password=''):
        if self._api_name == 'jqdata':
            return

        if username:
            import jqdatasdk
            jqdatasdk.auth(username, password)

    @property
    def api(self):
        if not hasattr(self, "_api"):
            raise NotImplementedError('api not specified')
        return self._api

    @lru_cache(2)
    def _get_trade_days(self, start_date=None, end_date=None):
        if start_date is not None:
            start_date = date2str(start_date)
        if end_date is not None:
            end_date = date2str(end_date)
        return list(self.api.get_trade_days(start_date=start_date,
                                            end_date=end_date))

    def _get_price(self, securities, start_date=None, end_date=None, count=None,
                   fields=None, skip_paused=False, fq='post'):
        start_date = date2str(start_date) if start_date is not None else None
        end_date = date2str(end_date) if end_date is not None else None
        if self._api_name == 'jqdata':
            if 'panel' in self.api.get_price.__code__.co_varnames:
                get_price = partial(self.api.get_price,
                                    panel=False,
                                    pre_factor_ref_date=end_date)
            else:
                get_price = partial(self.api.get_price,
                                    pre_factor_ref_date=end_date)
        else:
            get_price = self.api.get_price
        p = get_price(
            securities, start_date=start_date, end_date=end_date, count=count,
            fields=fields, skip_paused=skip_paused, fq=fq
        )
        if hasattr(p, 'to_frame'):
            p = p.to_frame()
            p.index.names = ['time', 'code']
            p.reset_index(inplace=True)
        return p

    def get_prices(self, securities, start_date=None, end_date=None,
                   period=None):
        if period is not None:
            trade_days = self._get_trade_days(start_date=end_date)
            if len(trade_days):
                end_date = trade_days[:period + 1][-1]
        p = self._get_price(
            fields=[self.price], securities=securities,
            start_date=start_date, end_date=end_date,
            fq=self.fq
        )
        p = p.set_index(['time', 'code'])[self.price].unstack('code').sort_index()
        return p

    def _get_industry(self, securities, start_date, end_date,
                      industry='jq_l1'):
        trade_days = self._get_trade_days(start_date, end_date)
        industries = map(partial(self.api.get_industry, securities), trade_days)

        industries = {
            d: {
                s: ind.get(s).get(industry, dict()).get('industry_name', 'NA')
                for s in securities
            }
            for d, ind in zip(trade_days, industries)
        }
        return pd.DataFrame(industries).T.sort_index()

    def get_groupby(self, securities, start_date, end_date):
        return self._get_industry(securities=securities,
                                  start_date=start_date, end_date=end_date,
                                  industry=self.industry)

    def _get_market_cap(self, securities, start_date, end_date, ln=False):
        trade_days = self._get_trade_days(start_date, end_date)

        query = self.api.query
        valuation = self.api.valuation

        def get_mkt_cap(s, date):
            if not s:
                return pd.Series(dtype='float64', name=date)
            elif len(s) > 3000:
                return get_mkt_cap(s[:3000], date).append(
                    get_mkt_cap(s[3000:], date)
                )
            q = query(valuation.code, valuation.market_cap).filter(
                valuation.code.in_(s)
            )
            return self.api.get_fundamentals(
                q, date=date2str(date)
            ).set_index('code').market_cap

        market_cap = pd.DataFrame(index=trade_days, columns=securities)
        market_cap = market_cap.apply(
            lambda s: get_mkt_cap(s.index.tolist(), s.name), axis=1
        ) * (10 ** 8)
        if ln:
            market_cap = np.log(market_cap)
        return market_cap

    def _get_circulating_market_cap(self, securities, start_date, end_date,
                                    ln=False):
        trade_days = self._get_trade_days(start_date, end_date)

        query = self.api.query
        valuation = self.api.valuation

        def get_circulating_cap(s, date):
            if not s:
                return pd.Series(dtype='float64', name=date)
            elif len(s) > 3000:
                return get_circulating_cap(s[:3000], date).append(
                    get_circulating_cap(s[3000:], date)
                )
            q = query(valuation.code, valuation.circulating_market_cap).filter(
                valuation.code.in_(s)
            )
            return self.api.get_fundamentals(
                q, date=date2str(date)
            ).set_index('code').circulating_market_cap

        cmarket_cap = pd.DataFrame(index=trade_days, columns=securities)
        cmarket_cap = cmarket_cap.apply(
            lambda s: get_circulating_cap(s.index.tolist(), s.name), axis=1
        ) * (10**8)
        if ln:
            cmarket_cap = np.log(cmarket_cap)
        return cmarket_cap

    def _get_average_weights(self, securities, start_date, end_date):
        return {sec: 1.0 for sec in securities}

    def get_weights(self, securities, start_date, end_date):
        start_date = date2str(start_date)
        end_date = date2str(end_date)

        if self.weight_method == 'avg':
            weight_api = self._get_average_weights
        elif self.weight_method == 'mktcap':
            weight_api = partial(self._get_market_cap, ln=False)
        elif self.weight_method == 'ln_mktcap':
            weight_api = partial(self._get_market_cap, ln=True)
        elif self.weight_method == 'cmktcap':
            weight_api = partial(self._get_circulating_market_cap, ln=False)
        elif self.weight_method == 'ln_cmktcap':
            weight_api = partial(self._get_circulating_market_cap, ln=True)
        else:
            raise ValueError('invalid weight_method')

        return weight_api(securities=securities, start_date=start_date,
                          end_date=end_date)

    @property
    def apis(self):
        return dict(prices=self.get_prices,
                    groupby=self.get_groupby,
                    weights=self.get_weights)

# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from fastcache import lru_cache
from functools import partial
import pyarrow.feather as feather

from .when import date2str, convert_date, today, now, Time, Date
from .factor_cache import save_factor_valeus_by_group, get_factor_values_by_cache, get_cache_dir


class DataApi(object):

    def __init__(self, price='close', fq='post',
                 industry='jq_l1', weight_method='avg', allow_cache=True, show_progress=True):
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
        allow_cache : 是否允许将分析必须数据以文件的形式缓存至本地, 默认允许, 缓存开启时, 首次加载耗时较长
        show_progress : 是否展示数据获取进度

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

        self.show_progress = show_progress
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

        valid_weight_method = ('avg', 'mktcap', 'ln_mktcap', 'cmktcap', 'ln_cmktcap')
        if weight_method in valid_weight_method:
            self.weight_method = weight_method
        else:
            raise ValueError("invalid 'weight_method' parameter, "
                             "should be one of %s" % str(valid_weight_method))
        self.ini_cache_cfg(allow_cache)

    @lru_cache(10)
    def get_ind_record(self, industry):
        mapping = self.api.get_industries(industry).to_dict()['name']
        ind_record = self.api.get_history_industry(industry).set_index('stock')
        ind_record['industry_name'] = ind_record['code'].map(mapping)
        ind_record.end_date = ind_record.end_date.fillna(Date(2040, 1, 1))
        return ind_record

    def ini_cache_cfg(self, allow_cache):
        self.allow_cache = allow_cache

        if self._api_name != 'jqdatasdk':
            self.allow_cache = False
        self.allow_industry_cache = False
        if self.allow_cache:
            if not self._api.is_auth():
                raise Exception("Please run jqdatasdk.auth first")
            privilege = self._api.get_privilege()
            if 'GET_HISTORY_INDUSTRY' in privilege:
                self.allow_industry_cache = True
            else:
                self.allow_industry_cache = False

            if 'FACTOR_BASICS' in privilege or 'GET_FACTOR_VALUES' in privilege:
                self.mkt_cache_api = 'factor'
            else:
                self.mkt_cache_api = 'valuation'

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
                   fields=None, skip_paused=False, fq='post', round=False):
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
            fields=fields, skip_paused=skip_paused, fq=fq, round=round
        )
        if hasattr(p, 'to_frame'):
            p = p.to_frame()
            p.index.names = ['time', 'code']
            p.reset_index(inplace=True)

        return p

    def _get_cached_price(self, securities, start_date=None, end_date=None, fq=None, overwrite=False):
        """获取缓存价格数据, 缓存文件中存储的数据是为未复权价格和后复权因子"""
        save_factor_valeus_by_group(start_date, end_date,
                                    factor_names='prices',
                                    overwrite=overwrite,
                                    show_progress=self.show_progress)
        trade_days = pd.to_datetime(self._get_trade_days(start_date, end_date))

        ret = []
        if self.show_progress:
            trade_days = tqdm(trade_days, desc="load price info : ")
        for day in trade_days:
            if day < today():
                p = get_factor_values_by_cache(
                    day, securities, factor_names='prices').reset_index()
            else:
                p = self.api.get_price(securities, start_date=day, end_date=day,
                                       skip_paused=False, round=False,
                                       fields=['open', 'close', 'factor'],
                                       fq='post', panel=False)
                p[['open', 'close']] = p[['open', 'close']].div(p['factor'], axis=0)
            p['time'] = day
            ret.append(p)
        ret = pd.concat(ret, ignore_index=True).sort_values(['code', 'time']).reset_index(drop=True)
        if fq == 'pre':
            # 前复权基准日期为最新一天
            latest_factor = self.api.get_price(securities,
                                               end_date=today(),
                                               count=1,
                                               skip_paused=False,
                                               round=False,
                                               fields=['factor'],
                                               fq='post',
                                               panel=False).set_index('code')
            ret = ret.set_index('code')
            ret.factor = ret.factor / latest_factor.factor
            ret = ret.reset_index().reindex(columns=['time', 'code', 'open', 'close', 'factor'])
        elif fq is None:
            ret.loc[ret['factor'].notna(), 'factor'] = 1.0
        ret[['open', 'close']] = ret[['open', 'close']].mul(ret['factor'], axis=0)
        return ret

    def get_prices(self, securities, start_date=None, end_date=None,
                   period=None):
        if period is not None:
            trade_days = self._get_trade_days(start_date=end_date)
            if len(trade_days):
                end_date = trade_days[:period + 1][-1]
        if self.allow_cache:
            p = self._get_cached_price(
                securities, start_date, end_date, fq=self.fq)
        else:
            p = self._get_price(
                fields=[self.price], securities=securities,
                start_date=start_date, end_date=end_date, round=False,
                fq=self.fq
            )
        p = p.set_index(['time', 'code'])[self.price].unstack('code').sort_index()
        return p

    def _get_industry(self, securities, start_date, end_date, industry='jq_l1'):
        trade_days = self._get_trade_days(start_date, end_date)
        industries = map(partial(self.api.get_industry, securities), trade_days)
        day_ind = zip(trade_days, industries)
        if self.show_progress:
            day_ind = tqdm(day_ind, desc='load industry info : ',
                           total=len(trade_days))
        industries = {
            d: {
                s: ind.get(s).get(industry, dict()).get('industry_name', 'NA')
                for s in securities
            }
            for d, ind in day_ind
        }
        return pd.DataFrame(industries).T.sort_index()

    def _get_cached_industry_one_day(self, date, securities=None, industry=None):
        date = convert_date(date)
        if self.allow_industry_cache:
            ind_record = self.get_ind_record(industry)
            if securities is not None:
                ind_record = ind_record[ind_record.index.isin(securities)]
            return ind_record[(ind_record.start_date <= date) & (date <= ind_record.end_date)].code
        else:
            ind_record = self.api.get_industry(securities, date, df=True)
            ind_record = ind_record[ind_record['type'] ==
                                    industry].set_index("code").industry_code
            return ind_record

    def _get_cached_industry(self, securities, start_date, end_date):
        ind_record = self.get_ind_record(self.industry)
        start_date = convert_date(start_date)
        end_date = convert_date(end_date)
        trade_days = self._get_trade_days(start_date, end_date)
        ind_record = ind_record[(ind_record.index.isin(securities))]
        if self.show_progress:
            trade_days = tqdm(trade_days, desc="load industry info : ")
        df_list = []
        for d in trade_days:
            rec = ind_record[(ind_record.start_date <= d) & (
                d <= ind_record.end_date)].industry_name
            rec.name = d
            df_list.append(rec)
        df = pd.DataFrame(df_list).reindex(columns=securities)
        return df.fillna('NA')

    def get_groupby(self, securities, start_date, end_date):
        if self.allow_industry_cache:
            return self._get_cached_industry(securities, start_date, end_date)
        else:
            return self._get_industry(securities=securities,
                                      start_date=start_date, end_date=end_date,
                                      industry=self.industry)

    def _get_cached_mkt_cap_by_valuation(self, securities, date, field, overwrite=False):
        """市值处理函数, 将获取的市值数据缓存到本地"""
        if not securities:
            return pd.Series(dtype='float64', name=date)

        query = self.api.query
        valuation = self.api.valuation
        cache_dir = os.path.join(get_cache_dir(), 'mkt_cap', date.strftime("%Y%m"))
        fp = os.path.join(cache_dir, date.strftime("%Y%m%d")) + '.feather'

        if os.path.exists(fp) and not overwrite:
            data = feather.read_feather(fp)
        else:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            codes = self.api.get_all_securities('stock').index.tolist()
            q = query(valuation.code,
                      valuation.market_cap,
                      valuation.circulating_market_cap).filter(
                          valuation.code.in_(codes))
            data = self.api.get_fundamentals(q, date=date2str(date))
            data[['market_cap', 'circulating_market_cap']] = data[
                ['market_cap', 'circulating_market_cap']] * (10 ** 8)
            if date < today() or (date == today() and now().time() >= Time(16, 30)):
                data.to_feather(fp)

        return data[data.code.isin(securities)].set_index('code')[field]

    def _get_market_cap(self, securities, start_date, end_date, ln=False, field='market_cap'):
        trade_days = self._get_trade_days(start_date, end_date)

        def get_mkt_cap(s, date, field):
            if not s:
                return pd.Series(dtype='float64', name=date)
            data = self.api.get_fundamentals(
                q, date=date2str(date)
            ).set_index('code')[field] * (10 ** 8)
            return data

        def get_mkt_cap_cache(s, date, field):
            cap = get_factor_values_by_cache(
                date, securities, factor_path=cache_dir).reindex(columns=[field])
            return cap[field]

        if self.allow_cache and len(trade_days) > 5:
            if self.mkt_cache_api == 'factor':
                desc = 'check/save cap cache :' if self.show_progress else False
                cache_dir = save_factor_valeus_by_group(start_date,
                                                        end_date,
                                                        factor_names=['market_cap', 'circulating_market_cap'],
                                                        group_name='mkt_cap',
                                                        show_progress=desc)
                market_api = get_mkt_cap_cache
            else:
                market_api = self._get_cached_mkt_cap_by_valuation
        else:
            market_api = get_mkt_cap
            query = self.api.query
            valuation = self.api.valuation

            if field == 'market_cap':
                q = query(valuation.code, valuation.market_cap).filter(
                    valuation.code.in_(securities))
            elif field == 'circulating_market_cap':
                q = query(valuation.code, valuation.circulating_market_cap).filter(
                    valuation.code.in_(securities))
            else:
                raise ValueError("不支持的字段 : {}".foramt(field))

        if self.show_progress:
            trade_days = tqdm(trade_days, desc="load cap info : ")

        market_cap = []
        for date in trade_days:
            cap = market_api(securities, date, field)
            cap.name = date
            market_cap.append(cap)
        market_cap = pd.concat(market_cap, axis=1).astype(float).reindex(index=securities)

        if ln:
            market_cap = np.log(market_cap)

        return market_cap.T

    def _get_circulating_market_cap(self, securities, start_date, end_date,
                                    ln=False):
        return self._get_market_cap(securities, start_date, end_date,
                                    ln=ln, field='circulating_market_cap')

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

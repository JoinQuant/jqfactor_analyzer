import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from jqfactor_analyzer.data import DataApi
from jqfactor_analyzer.factor_cache import save_factor_values_by_group, get_factor_values_by_cache
from jqfactor_analyzer.plot_utils import _use_chinese
from functools import lru_cache


dataapi = DataApi(allow_cache=True, show_progress=True)


def get_factor_style_returns(factors=None, start_date=None, end_date=None,
                             count=None, universe=None, industry='sw_l1'):
    if dataapi._api_name == 'jqdatasdk':
        func = dataapi.api.get_factor_style_returns
    else:
        import jqfactor
        func = jqfactor.get_factor_style_returns
    return func(factors=factors, start_date=start_date, end_date=end_date,
                count=count, universe=universe, industry=industry)


def get_price(security, start_date, end_date, fields):
    func = partial(dataapi.api.get_price, security=security,
                   start_date=start_date, end_date=end_date, fields=fields)
    if dataapi._api_name == 'jqdatasdk':
        return func()
    else:
        return func(pre_factor_ref_date=datetime.date.today())


def get_index_style_exposure(index, factors=None,
                             start_date=None, end_date=None, count=None):
    if dataapi._api_name == 'jqdatasdk':
        func = dataapi.api.get_index_style_exposure
    else:
        import jqfactor
        func = jqfactor.get_index_style_exposure
    return func(index=index, factors=factors,
                start_date=start_date, end_date=end_date, count=count)


class AttributionAnalysis():
    """归因分析

    用户需要提供的数据:
    1. 日度股票持仓权重 (加总不为 1 的剩余部分视为现金)
    2. 组合的的日度收益率 (使用 T 日持仓盘后的因子暴露与 T+1 日的收益进行归因分析)

    组合风格因子暴露 (含行业, country) = sum(组合权重 * 个股因子值), country 暴露为总的股票持仓权重
    组合风格收益率 (含行业, country) = sum(组合风格因子暴露 * factor_return)
    组合特异收益率 = 组合总收益率 - 组合风格收益率(含行业, country 或 cash)
    """

    def __init__(self, weights, daily_return,
                 style_type='style_pro', industry='sw_l1',
                 use_cn=True, show_data_progress=True):
        """
        参数
        ----------
            weights:
                持仓权重信息, index 是日期, columns 是标的代码, value 对应的是当天的仓位占比 (单日仓位占比总和不为 1 时, 剩余部分认为是当天的现金)
            daily_return:
                Series, index 是日期, values 为当天账户的收益率
            style_type:
                所选的风格因子类型, 'style' 和 'style_pro' 中的一个
            industry:
                行业分类, 可选: 'sw_l1' 或 'jq_l1'
            use_cn:
                绘图时是否使用中文
            show_data_progress:
                是否展示数据获取进度 (使用本地缓存, 第一次运行时速度较慢, 后续对于本地不存在的数据将增量缓存)

        所有属性列表
        ----------
            style_exposure:
                组合风格因子暴露
            industry_exposure:
                组合行业因子暴露
            exposure_portfolio:
                组合风格 / 行业及 country 的暴露
            attr_daily_return:
                组合归因日收益率
            attr_returns:
                组合归因累积收益汇总

        所有方法列表
        ----------
            get_exposure2bench(index_symbol):
                获取相对于指数的暴露
            get_attr_daily_returns2bench(index_symbol):
                获取相对于指数的日归因收益
            get_attr_returns2bench(index_symbol):
                获取相对于指数的累积归因收益

            plot_exposure(factors='style', index_symbol=None, use_cn=True, figsize=(15, 8))
                绘制风格或行业暴露, 当指定 index_symbol 时, 返回的是相对指数的暴露, 否则为组合自身的暴露
            plot_returns(factors='style', index_symbol=None, use_cn=True, figsize=(15, 8))
                绘制风格或者行业的暴露收益, 当指定 index_symbol 时, 返回的是相对指数的暴露收益, 否则为组合自身的暴露收益
            plot_exposure_and_returns(self, factors, index_symbol=None, use_cn=True, figsize=(12, 6))
                同时绘制暴露和收益信息
        """

        self.STYLE_TYPE_DICT = {
            'style': ['size', 'beta', 'momentum', 'residual_volatility', 'non_linear_size',
                      'book_to_price_ratio', 'liquidity', 'earnings_yield', 'growth', 'leverage'],
            'style_pro': ['btop', 'divyild', 'earnqlty', 'earnvar', 'earnyild', 'financial_leverage',
                          'invsqlty', 'liquidty', 'long_growth', 'ltrevrsl', 'market_beta', 'market_size',
                          'midcap', 'profit', 'relative_momentum', 'resvol']
        }
        weights.index = pd.to_datetime(weights.index)
        daily_return.index = pd.to_datetime(daily_return.index)
        weights.loc[weights.sum(axis=1) > 1] = weights.div(weights.sum(axis=1), axis=0)
        self.weights = weights.replace(0, np.nan)
        self.daily_return = daily_return
        self.style_factor_names = self.STYLE_TYPE_DICT[style_type]
        self.industry = industry
        self.industry_code = list(
            set(dataapi.api.get_industries(industry, date=weights.index[0]).index) |
            set(dataapi.api.get_industries(industry, date=weights.index[-1]).index)
        )
        self.style_type = style_type
        self.show_progress = show_data_progress
        self.factor_cache_directory = self.check_factor_values()

        # 当日收盘后的暴露
        self.style_exposure = self.calc_style_exposure()
        # 当日收盘后的暴露
        self.industry_exposure = self.calc_industry_exposure()
        # 当日收盘后的暴露
        self.exposure_portfolio = pd.concat([self.style_exposure, self.industry_exposure], axis=1)
        self.exposure_portfolio['country'] = self.weights.sum(axis=1)
        self.use_cn = use_cn
        if use_cn:
            _use_chinese(True)

        self._attr_daily_returns = None
        self._attr_returns = None
        self._factor_returns = None
        self._factor_cn_name = None

    def _get_factor_cn_name(self):
        """获取行业及风格因子的中文名称"""
        industry_info = dataapi.api.get_industries(self.industry).name
        factor_info = dataapi.api.get_all_factors()
        factor_info = factor_info[factor_info.category ==
                                  self.style_type].set_index("factor").factor_intro
        factor_info = pd.concat([industry_info, factor_info])
        factor_info['common_return'] = '因子收益'
        factor_info['specific_return'] = '特异收益'
        factor_info['total_return'] = '总收益'
        factor_info['cash'] = '现金'
        factor_info['country'] = 'country'
        self._factor_cn_name = factor_info
        return factor_info

    @property
    def factor_cn_name(self):
        if self._factor_cn_name is None:
            return self._get_factor_cn_name()
        else:
            return self._factor_cn_name

    def check_factor_values(self):
        """检查并缓存因子数据到本地"""
        start_date = self.weights.index[0]
        end_date = self.weights.index[-1]
        return save_factor_values_by_group(start_date, end_date,
                                           self.style_factor_names,
                                           show_progress=self.show_progress)

    def _get_style_exposure_daily(self, date, weight):
        weight = weight.dropna()
        resdaily = get_factor_values_by_cache(
            date,
            codes=weight.index,
            factor_names=self.style_factor_names,
            factor_path=self.factor_cache_directory).T
        resdaily = resdaily.mul(weight).sum(axis=1, min_count=1)
        resdaily.name = date
        return resdaily

    def calc_style_exposure(self):
        """计算组合的风格因子暴露
        返回: 一个 dataframe, index 为日期, columns 为风格因子名, values 为暴露值"""

        iters = self.weights.iterrows()

        if self.show_progress:
            iters = tqdm(iters, total=self.weights.shape[0], desc='calc_style_exposure ')
        results = []
        for date, weight in iters:
            results.append(self._get_style_exposure_daily(date, weight))
        return pd.DataFrame(results)

    def _get_industry_exposure_daily(self, date, weight):
        weight = weight.dropna()
        resdaily = pd.get_dummies(dataapi._get_cached_industry_one_day(
            str(date.date()), securities=weight.index, industry=self.industry))
        resdaily = resdaily.mul(weight, axis=0).sum(axis=0, min_count=1)
        resdaily.name = date
        return resdaily

    def calc_industry_exposure(self):
        """计算组合的行业因子暴露
        返回: 一个 dataframe, index 为日期, columns为风格因子名, values为暴露值"""
        iters = self.weights.iterrows()
        if self.show_progress:
            iters = tqdm(iters, total=self.weights.shape[0], desc='calc_industry_exposure ')
        results = []
        for date, weight in iters:
            results.append(self._get_industry_exposure_daily(date, weight))
        return pd.DataFrame(results).reindex(columns=self.industry_code, fill_value=0)

    @property
    def attr_daily_returns(self):
        if self._attr_daily_returns is None:
            return self.calc_attr_returns()[0]
        else:
            return self._attr_daily_returns

    @property
    def attr_returns(self):
        if self._attr_returns is None:
            return self.calc_attr_returns()[1]
        else:
            return self._attr_returns

    @property
    def factor_returns(self):
        if self._factor_returns is None:
            exposure_portfolio = self.exposure_portfolio.copy()
            self._factor_returns = get_factor_style_returns(
                exposure_portfolio.columns.tolist(),
                self.exposure_portfolio.index[0],
                dataapi.api.get_trade_days(self.exposure_portfolio.index[-1], count=2)[-1],
                industry=self.industry,
                universe='zzqz')
            return self._factor_returns
        else:
            return self._factor_returns

    @lru_cache()
    def _get_index_returns(self, index_symbol, start_date, end_date):
        index_return = get_price(index_symbol,
                                 start_date=start_date,
                                 end_date=end_date,
                                 fields='close')['close'].pct_change()
        return index_return

    @lru_cache()
    def _get_index_exposure(self, index_symbol):
        index_exposure = get_index_style_exposure(
            index_symbol,
            factors=self.style_exposure.columns.tolist() + self.industry_exposure.columns.tolist(),
            start_date=str(self.weights.index[0]),
            end_date=str(self.weights.index[-1]))
        index_exposure = index_exposure.mul(self.weights.sum(axis=1), axis=0)
        index_exposure['country'] = 1
        return index_exposure

    @lru_cache()
    def get_exposure2bench(self, index_symbol):
        """获取相对于指数的暴露"""
        index_exposure = self._get_index_exposure(index_symbol)
        return self.exposure_portfolio - index_exposure

    @lru_cache()
    def get_attr_daily_returns2bench(self, index_symbol):
        """获取相对于指数的日归因收益率
        返回: 一个 datafame, index 是日期, value 为对应日期的收益率值
        columns 为风格因子/行业因子/现金cash/因子总收益common_return(含风格,行业)/特异收益率 specific_return 及组合总收益率 total_return
        注意: 日收益率直接加总, 可能和实际的最终收益率不一致, 因为没考虑到资产的变动情况
        """
        exposure2bench = self.get_exposure2bench(index_symbol)
        exposure2bench = exposure2bench.reindex(self.factor_returns.index)

        index_return = self._get_index_returns(index_symbol,
                                               start_date=exposure2bench.index[0],
                                               end_date=exposure2bench.index[-1])
        daily_return = self.daily_return - index_return

        attr_daily_returns2bench = exposure2bench.shift()[1:].mul(self.factor_returns)
        # country 收益为 0, 无意义
        del attr_daily_returns2bench['country']
        attr_daily_returns2bench['common_return'] = attr_daily_returns2bench[self.style_exposure.columns.tolist() +
                                                                             self.industry_exposure.columns.tolist()].sum(axis=1)
        attr_daily_returns2bench['cash'] = index_return * exposure2bench.country.shift()
        attr_daily_returns2bench['specific_return'] = daily_return - \
            attr_daily_returns2bench['common_return'] - \
            attr_daily_returns2bench['cash']
        attr_daily_returns2bench['total_return'] = daily_return
        return attr_daily_returns2bench

    @lru_cache()
    def get_attr_returns2bench(self, index_symbol):
        """获取相对于指数的累积归因收益
        将超额收益分解成了:
        1.common_return (因子收益, 又可进一步拆分成风格和行业);
        2.cash (现金收益, 假设组合本身现金部分的收益为0, 则相对于指数的超额收益为"-1 * 指数收益");
              累积算法: (组合收益率 + 1).cumpord() - (日现金收益率+组合收益率 + 1).cumpord()
        3.specific_return: 残差, 无法被风格和行业因子解释的部分, 即为主动收益, 现金收益实际也可划分到主动收益中
        """
        index_return = self._get_index_returns(index_symbol,
                                               start_date=self.factor_returns.index[0],
                                               end_date=self.factor_returns.index[-1])

        attr_daily_returns2bench = self.get_attr_daily_returns2bench("000905.XSHG")
        # 假设持仓的现金用于购买指数时的净值
        position_with_cash_net = ((-attr_daily_returns2bench.cash + self.daily_return).fillna(0) + 1).cumprod()
        # 持仓本身的净值
        position_net = (self.daily_return.fillna(0) + 1).cumprod()
        # 假设指数满仓时的超额
        t_net = position_net - (index_return + 1).fillna(1).cumprod()
        # 假设指数调整仓位到和组合一致(风格暴露)的超额
        net = position_net - (index_return * self.weights.sum(axis=1).shift() + 1).fillna(1).cumprod()
        # 超额的暴露收益
        attr_returns2bench2 = attr_daily_returns2bench.mul(net.shift() + 1, axis=0).cumsum()
        # 现金的收益 = 持仓本身的净值 - 假设持仓的现金用于购买指数的净值
        attr_returns2bench2['cash'] = position_net - position_with_cash_net
        # 超额收益
        attr_returns2bench2['total_return'] = t_net
        # 风格 + 行业因子收益, 不含现金
        attr_returns2bench2['common_return'] = attr_returns2bench2[self.style_exposure.columns.tolist() +
                                                                   self.industry_exposure.columns.tolist()].sum(axis=1)
        attr_returns2bench2.loc[attr_returns2bench2.cash.isna(), 'common_return'] = np.nan
        # 除风格,现金以外的无法解释的收益
        attr_returns2bench2['specific_return'] = (
            attr_returns2bench2['total_return'] - attr_returns2bench2['common_return'] - attr_returns2bench2['cash']
        )
        return attr_returns2bench2

    def calc_attr_returns(self):
        """计算风格归因收益, country 收益率为国家收益 (这里的国家收益是用均衡大小市值后 (根号市值) 回归得到的"""
        self._attr_daily_returns = self.exposure_portfolio.reindex(
            self.factor_returns.index).shift(1).mul(self.factor_returns)
        self._attr_daily_returns['common_return'] = self._attr_daily_returns.sum(axis=1)
        self._attr_daily_returns['specific_return'] = self.daily_return - self._attr_daily_returns['common_return']
        self._attr_daily_returns['total_return'] = self.daily_return

        cum_return = (self._attr_daily_returns.total_return.fillna(0) + 1).cumprod()
        self._attr_returns = self._attr_daily_returns.mul(cum_return.shift(1), axis=0).cumsum()

        return self._attr_daily_returns, self._attr_returns

    def plot_data(self, data, title=None, figsize=(15, 8)):
        ax = data.plot(figsize=figsize, title=title)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

    def plot_exposure(self, factors='style', index_symbol=None, figsize=(15, 7)):
        """绘制风格暴露
        factors: 绘制的暴露类型 , 可选 'style'(所有风格因子), 'industry'(所有行业因子), 也可以传递一个list, list 为 exposure_portfolio 中 columns 的一个或者多个
        index_symbol: 基准指数代码, 指定时绘制相对于指数的暴露, 默认 None 为组合本身的暴露
        figsize: 画布大小
        """
        exposure = self.exposure_portfolio if index_symbol is None else self.get_exposure2bench(index_symbol)
        if isinstance(factors, str):
            if factors == 'style':
                exposure = exposure[self.style_exposure.columns]
            elif factors == 'industry':
                exposure = exposure[self.industry_exposure.columns]
            else:
                exposure = exposure[[factors]]
        else:
            exposure = exposure[factors]

        if self.use_cn:
            exposure = exposure.rename(columns=self.factor_cn_name)
            title = '组合相对{}暴露'.format(index_symbol) if index_symbol else '组合暴露'
        else:
            title = 'exposure of {}'.format(index_symbol) if index_symbol else 'exposure'
        self.plot_data(exposure, title=title, figsize=figsize)

    def plot_returns(self, factors='style', index_symbol=None, figsize=(15, 7)):
        """绘制归因分析收益信息
        factors: 绘制的暴露类型, 可选 'style'(所有风格因子), 'industry'(所有行业因子), 也可以传递一个 list, list 为 exposure_portfolio 中 columns 的一个或者多个
                同时也支持指定 ['common_return'(风格总收益), 'specific_return'(特异收益), 'total_return'(总收益),
                               'country'(国家因子收益,当指定index_symbol时会用现金相对于指数的收益替代)]
        index_symbol: 基准指数代码, 指定时绘制相对于指数的暴露, 默认 None 为组合本身的暴露
        figsize: 画布大小
        """
        returns = self.attr_returns if index_symbol is None else self.get_attr_returns2bench(index_symbol)
        if isinstance(factors, str):
            if factors == 'style':
                returns = returns[self.style_exposure.columns]
            elif factors == 'industry':
                returns = returns[self.industry_exposure.columns]
            else:
                if index_symbol and factors == 'country':
                    factors = 'cash'
                if factors not in returns.columns:
                    raise ValueError("错误的因子名称: {}".format(factors))
                returns = returns[[factors]]
        else:
            if index_symbol and 'country' in factors:
                factors = [x if x != 'country' else 'cash' for x in factors]
            wrong_factors = [x for x in factors if x not in returns.columns]
            if wrong_factors:
                raise ValueError("错误的因子名称: {}".format(wrong_factors))
            returns = returns[factors]

        if self.use_cn:
            returns = returns.rename(columns=self.factor_cn_name)
            title = "累积归因收益 (相对{})".format(
                index_symbol) if index_symbol else "累积归因收益"
        else:
            title = 'cum return to {}  '.format(
                index_symbol) if index_symbol else "cum return"
        self.plot_data(returns, title=title, figsize=figsize)

    def plot_exposure_and_returns(self, factors='style', index_symbol=None, show_factor_perf=False, figsize=(12, 6)):
        """将因子暴露与收益同时绘制在多个子图上
        factors: 绘制的暴露类型, 可选 'style'(所有风格因子) , 'industry'(所有行业因子), 也可以传递一个 list, list为 exposure_portfolio 中 columns 的一个或者多个
                 当指定 index_symbol 时, country 会用现金相对于指数的收益替代)
        index_symbol: 基准指数代码,指定时绘制相对于指数的暴露及收益 , 默认None为组合本身的暴露和收益
        show_factor_perf: 是否同时绘制因子表现
        figsize: 画布大小, 这里第一个参数是画布的宽度, 第二个参数为单个子图的高度
        """
        if isinstance(factors, str):
            if factors == 'style':
                factors = self.style_exposure.columns
            elif factors == 'industry':
                factors = self.industry_exposure.columns
            else:
                factors = [factors]
        if index_symbol:
            exposure = self.get_exposure2bench(index_symbol).rename(columns={"country": "cash"})
            returns = self.get_attr_returns2bench(index_symbol)
        else:
            exposure = self.exposure_portfolio
            returns = self.attr_returns
        exposure, returns = exposure.align(returns, join='outer')
        if show_factor_perf:
            factor_performance = self.factor_returns.cumsum().reindex(exposure.index)

        num_factors = len(factors)
        # 每行最多两个子图
        ncols = 2 if num_factors > 1 else 1
        nrows = (num_factors + 1) // ncols if num_factors > 1 else 1

        fixed_width, base_height_per_row = figsize
        height_per_row = base_height_per_row if ncols == 1 else base_height_per_row / 2
        total_height = max(1, nrows) * height_per_row

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fixed_width, total_height))
        axes = axes.flatten() if num_factors > 1 else [axes]

        # 删除多余的子图
        for j in range(len(factors), len(axes)):
            fig.delaxes(axes[j])

        for i, factor_name in enumerate(factors):
            if index_symbol and factor_name == 'country':
                factor_name = 'cash'
            if factor_name not in exposure.columns:
                raise ValueError("错误的因子名称: {}".format(factor_name))
            e = exposure[factor_name]
            r = returns[factor_name]

            ax1 = axes[i]
            e.plot(kind='area', stacked=False, alpha=0.5, ax=ax1, color='skyblue')

            ax2 = ax1.twinx()
            r.plot(ax=ax2, color='red')
            if factor_name != 'cash' and show_factor_perf:
                factor_performance[factor_name].plot(ax=ax2, color='blue')
            ax1.set_title(factor_name if not self.use_cn else self.factor_cn_name.get(factor_name))
        labels = ['暴露', '因子收益', '因子表现'] if self.use_cn else ['exposure', 'return', 'factor performance']
        fig.legend(labels[:1], loc='upper left')

        # 手动创建图例条目
        custom_lines = [Line2D([0], [0], color='red', lw=2),
                        Line2D([0], [0], color='blue', lw=2)]
        # 创建自定义图例
        fig.legend(custom_lines, labels[1:], loc='upper right',
                   bbox_to_anchor=(1, 1.02), bbox_transform=plt.gcf().transFigure)
        fig.suptitle('因子暴露与收益图' if self.use_cn else 'factor exposure and return', y=1.02)
        plt.tight_layout()
        plt.show()

    def plot_disable_chinese_label(self):
        """关闭中文图例显示

        画图时默认会从系统中查找中文字体显示以中文图例
        如果找不到中文字体则默认使用英文图例
        当找到中文字体但中文显示乱码时, 可调用此 API 关闭中文图例显示而使用英文
        """
        _use_chinese(False)
        self.use_cn = False

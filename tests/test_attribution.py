import os
import datetime
import pandas as pd
from functools import partial

from jqfactor_analyzer import AttributionAnalysis, DataApi

try:
    import jqdata
except:
    # 使用 sdk 进行测试时可能需要先登陆
    import jqdatasdk

weights = pd.read_csv(
    os.path.join(os.getcwd(), "jqfactor_analyzer/sample_data/weight_info.csv"), index_col=0)
returns = weights.pop("return")
index_weights = pd.read_csv(
    os.path.join(os.getcwd(), "jqfactor_analyzer/sample_data/index_weight_info.csv"), index_col=0)
index_returns = index_weights.pop("return")

dataapi = DataApi(allow_cache=True, show_progress=True)
w2 = index_weights.div(index_weights.sum(axis=1), axis=0) * 0.1
r2 = dataapi.api.get_price('000905.XSHG',
                           start_date='2020-01-01',
                           end_date='2024-07-01',
                           fields='close',
                           fq=None)['close'].pct_change() * 0.1
An = AttributionAnalysis(w2, r2, style_type='style' )
df = An.get_attr_returns2bench("000905.XSHG")


def test_get_attr_returns2bench():
    assert df.shape == (1088, 46)
    assert set(df.columns) == set([
        'beta', 'book_to_price_ratio', 'earnings_yield', 'growth', 'leverage',
        'liquidity', 'momentum', 'non_linear_size', 'residual_volatility',
        'size', '801750', '801160', '801200', '801780', '801050', '801040',
        '801960', '801170', '801760', '801790', '801720', '801130', '801080',
        '801110', '801890', '801140', '801120', '801180', '801880', '801030',
        '801770', '801740', '801730', '801950', '801010', '801230', '801710',
        '801970', '801210', '801150', '801020', '801980', 'common_return',
        'cash', 'specific_return', 'total_return']
    )


def test_net():
    func = partial(dataapi.api.get_price,
                   '000905.XSHG',
                   start_date='2020-01-01',
                   end_date='2024-07-01',
                   fields='close')
    if dataapi._api_name == 'jqdata':
        index_return = func(pre_factor_ref_date=datetime.date.today())['close'].pct_change()[1:]
    else:
        index_return = func()['close'].pct_change()[1:]
    index_net = (index_return.fillna(0) + 1).cumprod()
    assert len(index_net) == 1087

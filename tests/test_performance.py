# -*- coding: utf-8 -*-


import pytest
import pandas as pd
from numpy import nan, float64

from jqfactor_analyzer.prepare import get_clean_factor_and_forward_returns
from jqfactor_analyzer.performance import (
    factor_information_coefficient,
    factor_autocorrelation,
    mean_information_coefficient,
    quantile_turnover,
    factor_returns, factor_alpha_beta,
    average_cumulative_return_by_quantile
)
from jqfactor_analyzer.utils import get_forward_returns_columns


dr = pd.date_range(start='2015-1-1', end='2015-1-2')
dr.name = 'date'
tickers = ['A', 'B', 'C', 'D']
factor = pd.DataFrame(index=dr,
                      columns=tickers,
                      data=[[1, 2, 3, 4],
                            [4, 3, 2, 1]]).stack()
factor.index = factor.index.set_names(['date', 'asset'])
factor.name = 'factor'
factor_data = pd.DataFrame()
factor_data['factor'] = factor
factor_data['group'] = pd.Series(index=factor.index,
                                 data=[1, 1, 2, 2, 1, 1, 2, 2],)
factor_data['weights'] = pd.Series(range(8), index=factor.index,
                                   dtype=float64) + 1


@pytest.mark.parametrize(
    ('factor_data', 'forward_returns', 'group_adjust',
     'by_group', 'expected_ix', 'expected_ic_val'),
    [(factor_data, [4, 3, 2, 1, 1, 2, 3, 4], False, False, dr, [-1., -1.]),
     (factor_data, [1, 2, 3, 4, 4, 3, 2, 1], False, False, dr, [1., 1.]),
     (factor_data, [1, 2, 3, 4, 4, 3, 2, 1], False, True,
      pd.MultiIndex.from_product([dr, [1, 2]], names=['date', 'group']),
      [1., 1., 1., 1.]),
     (factor_data, [1, 2, 3, 4, 4, 3, 2, 1], True, True,
      pd.MultiIndex.from_product([dr, [1, 2]], names=['date', 'group']),
      [1., 1., 1., 1.])]
)
def test_information_coefficient(factor_data,
                                 forward_returns,
                                 group_adjust,
                                 by_group,
                                 expected_ix,
                                 expected_ic_val):

    factor_data = factor_data.copy()
    factor_data['period_1'] = pd.Series(index=factor_data.index,
                                        data=forward_returns)

    ic = factor_information_coefficient(factor_data=factor_data,
                                        group_adjust=group_adjust,
                                        by_group=by_group)

    expected_ic_df = pd.DataFrame(index=expected_ix,
                                  columns=pd.Index(['period_1'], dtype='object'),
                                  data=expected_ic_val)

    pd.testing.assert_frame_equal(ic, expected_ic_df)

@pytest.mark.parametrize(
    (
        'factor_data', 'forward_returns', 'group_adjust',
        'by_group', 'by_time', 'expected_ix', 'expected_ic_val'
    ), [
        (factor_data, [4, 3, 2, 1, 1, 2, 3, 4], False, False, 'D',
         dr, [-1., -1.]),
        (factor_data, [1, 2, 3, 4, 4, 3, 2, 1], False, False, 'W',
         pd.DatetimeIndex(['2015-01-04'], name='date', freq='W-SUN'), [1.]),
        (factor_data, [1, 2, 3, 4, 4, 3, 2, 1], False, True, None,
         pd.Int64Index([1, 2], name='group'), [1., 1.]),
        (factor_data, [1, 2, 3, 4, 4, 3, 2, 1], False, True, 'W',
         pd.MultiIndex.from_product(
             [pd.DatetimeIndex(['2015-01-04'], name='date', freq='W-SUN'),
              [1, 2]],
             names=['date', 'group']
         ),
         [1., 1.])
    ]
)
def test_mean_information_coefficient(factor_data,
                                      forward_returns,
                                      group_adjust,
                                      by_group,
                                      by_time,
                                      expected_ix,
                                      expected_ic_val):

    factor_data = factor_data.copy()
    factor_data['period_1'] = pd.Series(index=factor_data.index,
                                        data=forward_returns)

    ic = mean_information_coefficient(factor_data,
                                      group_adjust=group_adjust,
                                      by_group=by_group,
                                      by_time=by_time)

    expected_ic_df = pd.DataFrame(index=expected_ix,
                                  columns=pd.Index(['period_1']),
                                  data=expected_ic_val)

    pd.testing.assert_frame_equal(ic, expected_ic_df,
                                  check_index_type=False,
                                  check_column_type=False)


@pytest.mark.parametrize(
    ('quantile_values', 'test_quantile', 'expected_vals'),
    [([[1.0, 2.0, 3.0, 4.0],
       [4.0, 3.0, 2.0, 1.0],
       [1.0, 2.0, 3.0, 4.0],
       [1.0, 2.0, 3.0, 4.0]],
      4.0,
      [nan, 1.0, 1.0, 0.0]),
     ([[1.0, 2.0, 3.0, 4.0],
       [1.0, 2.0, 3.0, 4.0],
       [1.0, 2.0, 3.0, 4.0],
       [1.0, 2.0, 3.0, 4.0]],
      3.0,
      [nan, 0.0, 0.0, 0.0]),
     ([[1.0, 2.0, 3.0, 4.0],
       [4.0, 3.0, 2.0, 1.0],
       [1.0, 2.0, 3.0, 4.0],
       [4.0, 3.0, 2.0, 1.0]],
      2.0,
      [nan, 1.0, 1.0, 1.0])]
)
def test_quantile_turnover(quantile_values, test_quantile,
                           expected_vals):

    dr = pd.date_range(start='2015-1-1', end='2015-1-4')
    dr.name = 'date'
    tickers = ['A', 'B', 'C', 'D']

    quantized_test_factor = pd.Series(
        pd.DataFrame(index=dr, columns=tickers, data=quantile_values).stack()
    )
    quantized_test_factor.index = quantized_test_factor.index.set_names(
        ['date', 'asset']
    )

    to = quantile_turnover(quantized_test_factor, test_quantile)

    expected = pd.Series(
        index=quantized_test_factor.index.levels[0], data=expected_vals)
    expected.name = test_quantile

    pd.testing.assert_series_equal(to, expected)


@pytest.mark.parametrize(
    ('factor_data', 'factor_vals', 'fwd_return_vals',
     'group_adjust', 'expected_vals'),
    [(factor_data, [1, 2, 3, 4, 4, 3, 2, 1], [4, 3, 2, 1, 1, 2, 3, 4],
      False, [-1.25000, -1.25000]),
     (factor_data, [1, 1, 1, 1, 1, 1, 1, 1], [4, 3, 2, 1, 1, 2, 3, 4],
      False, [0.0, 0.0]),
     (factor_data, [1, 2, 3, 4, 4, 3, 2, 1], [4, 3, 2, 1, 1, 2, 3, 4],
      True, [-0.5, -0.5]),
     (factor_data, [1, 2, 3, 4, 1, 2, 3, 4], [1, 4, 1, 2, 1, 2, 2, 1],
      True, [1.0, 0.0]),
     (factor_data, [1, 1, 1, 1, 1, 1, 1, 1], [4, 3, 2, 1, 1, 2, 3, 4],
      True, [0.0, 0.0])]
)
def test_factor_returns(factor_data,
                        factor_vals,
                        fwd_return_vals,
                        group_adjust,
                        expected_vals):

    factor_data = factor_data.copy()
    factor_data['period_1'] = fwd_return_vals
    factor_data['factor'] = factor_vals

    factor_returns_s = factor_returns(factor_data=factor_data,
                                      demeaned=True,
                                      group_adjust=group_adjust)

    expected = pd.DataFrame(
        index=dr,
        data=expected_vals,
        columns=get_forward_returns_columns(factor_data.columns)
    )

    pd.testing.assert_frame_equal(factor_returns_s, expected)


@pytest.mark.parametrize(
    ('factor_data', 'fwd_return_vals', 'alpha', 'beta'),
    [(factor_data, [1, 2, 3, 4, 1, 1, 1, 1], -1, 5. / 6.)]
)
def test_factor_alpha_beta(factor_data, fwd_return_vals, alpha, beta):

    factor_data = factor_data.copy()
    factor_data['period_1'] = fwd_return_vals

    ab = factor_alpha_beta(factor_data=factor_data)

    expected = pd.DataFrame(columns=['period_1'],
                            index=['Ann. alpha', 'beta'],
                            data=[alpha, beta])

    pd.testing.assert_frame_equal(ab, expected)

@pytest.mark.parametrize(
    ('factor_values', 'end_date', 'period', 'expected_vals'),
    [([[1.0, 2.0, 3.0, 4.0],
       [1.0, 2.0, 3.0, 4.0],
       [1.0, 2.0, 3.0, 4.0],
       [1.0, 2.0, 3.0, 4.0]],
      '2015-1-4', 1,
      [nan, 1.0, 1.0, 1.0]),
     ([[4.0, 3.0, 2.0, 1.0],
       [1.0, 2.0, 3.0, 4.0],
       [4.0, 3.0, 2.0, 1.0],
       [1.0, 2.0, 3.0, 4.0]],
      '2015-1-4', 1,
      [nan, -1.0, -1.0, -1.0]),
     ([[1.0, 2.0, 3.0, 4.0],
       [2.0, 1.0, 4.0, 3.0],
       [4.0, 3.0, 2.0, 1.0],
       [1.0, 2.0, 3.0, 4.0],
       [2.0, 1.0, 4.0, 3.0],
       [4.0, 3.0, 2.0, 1.0],
       [2.0, 1.0, 4.0, 3.0],
       [4.0, 3.0, 2.0, 1.0],
       [1.0, 2.0, 3.0, 4.0],
       [2.0, 1.0, 4.0, 3.0],
       [2.0, 1.0, 4.0, 3.0],
       [4.0, 3.0, 2.0, 1.0]],
      '2015-1-12', 3,
      [nan, nan, nan, 1.0, 1.0, 1.0, 0.6, -0.6, -1.0, 1.0, -0.6, -1.0])]
)
def test_factor_autocorrelation(factor_values,
                                end_date,
                                period,
                                expected_vals):
    dr = pd.date_range(start='2015-1-1', end=end_date)
    dr.name = 'date'
    tickers = ['A', 'B', 'C', 'D']
    factor = pd.DataFrame(index=dr,
                          columns=tickers,
                          data=factor_values).stack()
    factor.index = factor.index.set_names(['date', 'asset'])

    factor_df = pd.DataFrame()
    factor_df['factor'] = factor

    fa = factor_autocorrelation(factor_df, period)
    expected = pd.Series(index=dr, data=expected_vals)
    expected.name = period

    pd.testing.assert_series_equal(fa, expected)

@pytest.mark.parametrize(
    ('before', 'after', 'demeaned', 'quantiles', 'expected_vals'),
    [(1, 2, False, 4,
      [[1.00, 0.0, -0.50, -0.75],
       [0.0, 0.0, 0.0, 0.0],
       [0.00, 0.00, 0.00, 0.00],
       [0.0, 0.0, 0.0, 0.0],
       [-0.20, 0.0, 0.25, 0.5625],
       [0.0, 0.0, 0.0, 0.0],
       [-0.3333333, 0.0, 0.50, 1.25],
       [0.0, 0.0, 0.0, 0.0]]),
     (1, 2, True, 4,
      [[0.8833333, 0.0, -0.5625, -1.015625],
       [0.0, 0.0, 0.0, 0.0],
       [-0.1166667, 0.0, -0.0625, -0.265625],
       [0.0, 0.0, 0.0, 0.0],
       [-0.3166667, 0.0, 0.1875, 0.296875],
       [0.0, 0.0, 0.0, 0.0],
       [-0.4500000, 0.0, 0.4375, 0.984375],
       [0.0, 0.0, 0.0, 0.0]]),
     (3, 0, False, 4,
      [[7.0, 3.0, 1.0, 0.0],
       [0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0],
       [-0.488, -0.36, -0.2, 0.0],
       [0.0, 0.0, 0.0, 0.0],
       [-0.703704, -0.55555555, -0.333333333, 0.0],
       [0.0, 0.0, 0.0, 0.0]]),
     (0, 3, True, 4,
      [[0.0, -0.5625, -1.015625, -1.488281],
       [0.0, 0.0, 0.0, 0.0],
       [0.0, -0.0625, -0.265625, -0.613281],
       [0.0, 0.0, 0.0, 0.0],
       [0.0, 0.1875, 0.296875, 0.339844],
       [0.0, 0.0, 0.0, 0.0],
       [0.0, 0.4375, 0.984375, 1.761719],
       [0.0, 0.0, 0.0, 0.0]]),
     (3, 3, False, 2,
      [[3.5, 1.5, 0.5, 0.0, -0.25, -0.375, -0.4375],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [-0.595852, -0.457778, -0.266667, 0.0, 0.375, 0.90625, 1.664062],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
     (3, 3, True, 2,
      [[2.047926, 0.978888, 0.383333, 0.0, -0.3125, -0.640625, -1.050781],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       [-2.047926, -0.978888, -0.383333, 0.0, 0.3125, 0.640625, 1.050781],
       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])]
)
def test_average_cumulative_return_by_quantile(before, after,
                                               demeaned, quantiles,
                                               expected_vals):
    dr = pd.date_range(start='2015-1-15', end='2015-2-1')
    dr.name = 'date'
    tickers = ['A', 'B', 'C', 'D']
    r1, r2, r3, r4 = (1.25, 1.50, 1.00, 0.50)
    data = [[r1**i, r2**i, r3**i, r4**i] for i in range(1, 19)]
    prices = pd.DataFrame(index=dr, columns=tickers, data=data)
    dr2 = pd.date_range(start='2015-1-21', end='2015-1-26')
    dr2.name = 'date'
    factor = pd.DataFrame(
        index=dr2, columns=tickers, data=[
            [3, 4, 2, 1],
            [3, 4, 2, 1],
            [3, 4, 2, 1],
            [3, 4, 2, 1],
            [3, 4, 2, 1],
            [3, 4, 2, 1]]).stack()

    factor_data = get_clean_factor_and_forward_returns(
        factor, prices, quantiles=quantiles, periods=range(0, after + 1)
    )

    avgrt = average_cumulative_return_by_quantile(
        factor_data, prices, before, after, demeaned)
    arrays = []
    for q in range(1, quantiles + 1):
        arrays.append((q, 'mean'))
        arrays.append((q, 'std'))
    index = pd.MultiIndex.from_tuples(arrays, names=['factor_quantile', None])
    expected = pd.DataFrame(
        index=index, columns=range(-before, after + 1), data=expected_vals)
    pd.testing.assert_frame_equal(avgrt, expected)

@pytest.mark.parametrize(
    ('before', 'after', 'demeaned', 'quantiles', 'expected_vals'),
    [(0, 2, False, 4,
      [[0.0, -0.50, -0.75],
       [0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0],
       [0.0, 0.25, 0.5625],
       [0.0, 0.0, 0.0],
       [0.0, 0.50, 1.25],
       [0.0, 0.0, 0.0]]),
     (0, 3, True, 4,
      [[0.0, -0.5625, -1.015625, -1.488281],
       [0.0, 0.0, 0.0, 0.0],
       [0.0, -0.0625, -0.265625, -0.613281],
       [0.0, 0.0, 0.0, 0.0],
       [0.0, 0.1875, 0.296875, 0.339844],
       [0.0, 0.0, 0.0, 0.0],
       [0.0, 0.4375, 0.984375, 1.761719],
       [0.0, 0.0, 0.0, 0.0]]),
     (0, 3, False, 2,
      [[0.0, -0.25, -0.375, -0.4375],
       [0.0, 0.0, 0.0, 0.0],
       [0.0, 0.375, 0.90625, 1.664062],
       [0.0, 0.0, 0.0, 0.0]]),
     (0, 3, True, 2,
      [[0.0, -0.3125, -0.640625, -1.050781],
       [0.0, 0.0, 0.0, 0.0],
       [0.0, 0.3125, 0.640625, 1.050781],
       [0.0, 0.0, 0.0, 0.0]])]
)
def test_average_cumulative_return_by_quantile_2(before, after,
                                                 demeaned, quantiles,
                                                 expected_vals):
    """Test varying factor asset universe

    at different dates there might be different assets
    """
    dr = pd.date_range(start='2015-1-15', end='2015-1-25')
    dr.name = 'date'
    tickers = ['A', 'B', 'C', 'D', 'E', 'F']
    r1, r2, r3, r4 = (1.25, 1.50, 1.00, 0.50)
    data = [[r1**i, r2**i, r3**i, r4**i, r2**i, r3**i]
            for i in range(1, 12)]
    prices = pd.DataFrame(index=dr, columns=tickers, data=data)
    dr2 = pd.date_range(start='2015-1-18', end='2015-1-21')
    dr2.name = 'date'
    factor = pd.DataFrame(index=dr2, columns=tickers,
                          data=[[3, 4, 2, 1, nan, nan],
                                [3, 4, 2, 1, nan, nan],
                                [3, nan, nan, 1, 4, 2],
                                [3, nan, nan, 1, 4, 2]]).stack()

    factor_data = get_clean_factor_and_forward_returns(
        factor, prices, quantiles=quantiles, periods=range(0, after + 1),
    )

    avgrt = average_cumulative_return_by_quantile(
        factor_data, prices, before, after, demeaned
    )
    arrays = []
    for q in range(1, quantiles + 1):
        arrays.append((q, 'mean'))
        arrays.append((q, 'std'))
    index = pd.MultiIndex.from_tuples(arrays, names=['factor_quantile', None])
    expected = pd.DataFrame(
        index=index, columns=range(-before, after + 1), data=expected_vals
    )
    pd.testing.assert_frame_equal(avgrt, expected)


# **API文档**

## 因子分析函数

```python
analyze_factor(factor, industry='jq_l1', quantiles=5, periods=(1, 5, 10), weight_method='avg', max_loss=0.25)
```

单因子分析函数



**参数**

* `factor`: 因子值，pandas.DataFrame格式的数据

   - index为日期，格式为pandas日期通用的DatetimeIndex，转换方法见[将自有因子值转换成 DataFrame 格式的数据](#将自有因子值转换成-dataframe-格式的数据)
   - columns为股票代码，格式要求符合聚宽的代码定义规则（如：平安银行的股票代码为000001.XSHE）
       - 如果是深交所上市的股票，在股票代码后面需要加入.XSHE
       - 如果是上交所上市的股票，在股票代码后面需要加入.XSHG

* `industry`: 行业分类, 默认为 `'jq_l1'`

  * `'sw_l1'`: 申万一级行业
  * `'sw_l2'`: 申万二级行业
  * `'sw_l3'`: 申万三级行业
  * `'jq_l1'`: 聚宽一级行业
  * `'jq_l2'`: 聚宽二级行业
  * `'zjw'`: 证监会行业

* `quantiles`: 分位数数量, 默认为 `5`

  `int`

  在因子分组中按照因子值大小平均分组的组数.

* `periods`: 调仓周期, 默认为 [1, 5, 10]

  `int` or `list[int]`

* `weight_method`: 基于分位数收益时的加权方法, 默认为 `'avg'`

  * `'avg'`: 等权重
  * `'mktcap'`：按总市值加权
  * `'ln_mktcap'`: 按总市值的对数加权
  * `'cmktcap'`: 按流通市值加权
  * `'ln_cmktcap'`: 按流通市值的对数加权

* `max_loss`: 因重复值或nan值太多而无效的因子值的最大占比, 默认为 0.25

  `float`

  允许的丢弃因子数据的最大百分比 (0.00 到 1.00),

  计算比较输入因子索引中的项目数和输出 DataFrame 索引中的项目数.

  因子数据本身存在缺陷 (例如 NaN),

  没有提供足够的价格数据来计算所有因子值的远期收益,

  或者因为分组失败, 因此可以部分地丢弃因子数据



**示例**

```python
#载入函数库
import pandas as pd
import jqfactor_analyzer as ja

# 获取 jqdatasdk 授权
# 输入用户名、密码，申请地址：http://t.cn/EINDOxE
# 聚宽官网及金融终端，使用方法参见：http://t.cn/EINcS4j
import jqdatasdk
jqdatasdk.auth('username', 'password')

# 对因子进行分析
far = ja.analyze_factor(
    factor_data,  # factor_data 为因子值的 pandas.DataFrame
    quantiles=10,
    periods=(1, 10),
    industry='jq_l1',
    weight_method='avg',
    max_loss=0.1
)

# 生成统计图表
far.create_full_tear_sheet(
    demeaned=False, group_adjust=False, by_group=False,
    turnover_periods=None, avgretplot=(5, 15), std_bar=False
)
```







### 绘制结果

#### 展示全部分析

```
far.create_full_tear_sheet(demeaned=False, group_adjust=False, by_group=False,
turnover_periods=None, avgretplot=(5, 15), std_bar=False)
```

**参数:**

- demeaned:
    - True: 对每天的因子收益和标准差去均值
    - False: 按分位数分组加权因子收益和标准差
- group_adjust:
    - True: 每天的因子收益和标准差按行业去均值
    - False: 按分位数分组加权因子收益和标准差
- by_group:
    - True: 按行业展示
    - False: 不按行业展示
- turnover_periods: 调仓周期
- avgretplot: tuple 因子预测的天数-(计算过去的天数, 计算未来的天数)
- std_bar:
    - True: 显示标准差
    - False: 不显示标准差

#### 因子值特征分析

```
far.create_summary_tear_sheet(demeaned=False, group_adjust=False)
```

**参数:**

- demeaned:
    - True: 对每日因子收益去均值求得因子收益表
    - False: 因子收益表
- group_adjust:
    - True: 按行业对因子收益去均值后求得因子收益表
    - False: 因子收益表

#### 因子收益分析

```
far.create_returns_tear_sheet(demeaned=False, group_adjust=False, by_group=False)

```

**参数:**

- demeaned:
    - True: 对每日因子收益去均值求得因子收益表
    - False: 因子收益表
- group_adjust:
    - True: 按行业对因子收益去均值后求得因子收益表
    - False: 因子收益表
- by_group:
    - True: 画各行业的各分位数平均收益图
- False: 不画各行业的各分位数平均收益图

#### 因子 IC 分析

```
far.create_information_tear_sheet(group_adjust=False, by_group=False)

```

**参数:**

- group_adjust:
    - True: 按行业对因子收益去均值后求得因子收益表
    - False: 因子收益表
- by_group:
    - True: 画按行业分组信息比率(IC)图
    - False: 画月度信息比率(IC)图

#### 因子换手率分析

```
far.create_turnover_tear_sheet(turnover_periods=None)

```

**参数:**

- turnover_periods: 调仓周期

#### 因子预测能力分析

```
far.create_event_returns_tear_sheet(avgretplot=(5, 15),demeaned=False, group_adjust=False,std_bar=False)

```

**参数:**

- avgretplot: tuple 因子预测的天数-(计算过去的天数, 计算未来的天数)
- demeaned:
    - True: 对每天的因子收益和标准差去均值
    - False: 按分位数分组加权因子收益和标准差
- group_adjust:
    - True: 每天的因子收益和标准差按行业去均值
    - False: 按分位数分组加权因子收益和标准差
- std_bar:
    - True: 显示标准差
    - False: 不显示标准差

#### 打印因子收益表

```
far.plot_returns_table(demeaned=False, group_adjust=False)

```

**参数：**

- demeaned:
    - True：对每日因子收益去均值求得因子收益表
    - False：因子收益表
- group_adjust:
    - True：按行业对因子收益去均值后求得因子收益表
    - False：因子收益表

#### 打印换手率表

```
far.plot_turnover_table()

```

#### 打印信息比率（IC）相关表

```
far.plot_information_table(group_adjust=False, method='rank')

```

**参数：**

- group_adjust:
    - True：按行业对因子收益去均值后求得因子收益用于计算IC
    - False：因子收益用于计算IC
- method：
    - 'rank'：用秩相关系数计算IC值
    - 'normal':用相关系数计算IC值

#### 打印个分位数统计表

```
far.plot_quantile_statistics_table()

```

#### 画信息比率(IC)时间序列图

```
far.plot_ic_ts(group_adjust=False, method='rank')

```

**参数：**

- group_adjust:
    - True：按行业对因子收益去均值后求得因子收益用于计算IC
    - False：因子收益用于计算IC
- method：
    - 'rank'：用秩相关系数计算IC值
    - 'normal':用相关系数计算IC值

#### 画信息比率分布直方图

```
far.plot_ic_hist(group_adjust=False, method='rank')

```

**参数：**

- group_adjust:
    - True：按行业对因子收益去均值后求得因子收益用于计算IC
    - False：因子收益用于计算IC
- method：
    - 'rank'：用秩相关系数计算IC值
    - 'normal':用相关系数计算IC值

#### 画信息比率 qq 图

```
far.plot_ic_qq(group_adjust=False, method='rank', theoretical_dist='norm')

```

**参数：**

- group_adjust:
    - True：按行业对因子收益去均值后求得因子收益用于计算IC
    - False：因子收益用于计算IC
- method：
    - 'rank'：用秩相关系数计算IC值
    - 'normal':用相关系数计算IC值
- theoretical_dist：
    - 'norm'：正态分布
    - 't'：t分布

#### 画各分位数平均收益图

```
far.plot_quantile_returns_bar(by_group=False, demeaned=False, group_adjust=False)

```

**参数：**

- by_group：
    - True：各行业的各分位数平均收益图
    - False：各分位数平均收益图
- demeaned:
    - True：对每日因子收益去均值后的各分位数平均收益图
    - False：各分位数平均收益图
- group_adjust:
    - True：按行业对因子收益去均值后的各分位数平均收益图
    - False：各分位数平均收益图

#### 画最高分位减最低分位收益图

```
far.plot_mean_quantile_returns_spread_time_series(demeaned=False, group_adjust=False, bandwidth=1)

```

**参数：**

- demeaned:
    - True：对每日因子收益去均值后的最高分位减最低分位收益图
    - False：最高分位减最低分位收益图
- group_adjust:
    - True：按行业对因子收益去均值后的最高分位减最低分位收益图图
    - False：最高分位减最低分位收益图
- bandwidth：n，加减n倍当日标准差

#### 画按行业分组信息比率(IC)图

```
far.plot_ic_by_group(group_adjust=False, method='rank')

```

**参数：**

- group_adjust:
    - True：按行业对因子收益去均值后的行业分组信息比率图
    - False：行业分组信息比率图
- method：
    - 'rank'：用秩相关系数计算IC值
    - 'normal':用相关系数计算IC值

#### 画因子自相关图

```
far.plot_factor_auto_correlation(rank=True)

```

**参数：**

- rank：
    - True：用秩相关系数
    - False：用相关系数

#### 画最高最低分位换手率图

```
far.plot_top_bottom_quantile_turnover(periods=(1, 3, 9))

```

**参数：**

- periods：调仓周期

#### 画月度信息比率(IC)图

```
far.plot_monthly_ic_heatmap(group_adjust=False)

```

**参数：**

- group_adjust:
    - True：按行业对因子收益去均值后的月度信息比率图
    - False：月度信息比率图

#### 画按因子值加权多空组合每日累积收益图

```
far.plot_cumulative_returns(period=1, demeaned=False, group_adjust=False)

```

**参数：**

- periods：调仓周期
- demeaned:
    - True：对每日因子收益去均值后的按因子值加权多空组合每日累积收益图
    - False：按因子值加权多空组合每日累积收益图
- group_adjust:
    - True：按行业对因子收益去均值后的按因子值加权多空组合每日累积收益图
    - False：按因子值加权多空组合每日累积收益图

#### 画做多最高分位做空最低分位多空组合每日累计收益图

```
far.plot_cumulative_returns_by_quantile(period=(1, 3, 9), demeaned=False, group_adjust=False)

```

**参数：**

- periods：调仓周期
- demeaned:
    - True：对每日因子收益去均值后的多空组合每日累计收益图
    - False：多空组合每日累计收益图
- group_adjust:
    - True：按行业对因子收益去均值后的多空组合每日累计收益图
    - False：多空组合每日累计收益图

#### 因子预测能力平均累计收益图

```
far.plot_quantile_average_cumulative_return(by_quantile=False, std_bar=False)

```

**参数：**

- by_quantile：
    - True：各分位数分别显示因子预测能力平均累计收益图
    - False：因子预测能力平均累计收益图
- std_bar：
    - True：显示标准差
    - False：不显示标准差

#### 画有效因子数量统计图

```
far.plot_events_distribution(num_days=1)

```

**参数：**

- num_days：统计间隔天数

#### 关闭中文图例显示

```
far.plot_disable_chinese_label()

```



### 属性列表

用于访问因子分析的结果，大部分为惰性属性，在访问才会计算结果并返回



#### 查看因子值

```
far.factor_data
```

- 类型：pandas.Series
- index：为日期和股票代码的MultiIndex

#### 去除 nan/inf，整理后的因子值、forward_return 和分位数

```
far.clean_factor_data
```

- 类型：pandas.DataFrame index：为日期和股票代码的MultiIndex
- columns：根据period选择后的forward_return(如果调仓周期为1天，那么forward_return为[第二天的收盘价-今天的收盘价]/今天的收盘价)、因子值、行业分组、分位数数组、权重

#### 按分位数分组加权平均因子收益

```
far.mean_return_by_quantile
```

- 类型：pandas.DataFrame
- index：分位数分组
- columns：调仓周期

#### 按分位数分组加权因子收益标准差

```
far.mean_return_std_by_quantile
```

- 类型：pandas.DataFrame
- index：分位数分组
- columns：调仓周期

#### 按分位数及日期分组加权平均因子收益

```
far.mean_return_by_date
```

- 类型：pandas.DataFrame
- index：为日期和分位数的MultiIndex
- columns：调仓周期

#### 按分位数及日期分组加权因子收益标准差

```
far.mean_return_std_by_date
```

- 类型：pandas.DataFrame
- index：为日期和分位数的MultiIndex
- columns：调仓周期

#### 按分位数及行业分组加权平均因子收益

```
far.mean_return_by_group
```

- 类型：pandas.DataFrame
- index：为行业和分位数的MultiIndex
- columns：调仓周期

#### 按分位数及行业分组加权因子收益标准差

```
far.mean_return_std_by_group
```

- 类型：pandas.DataFrame
- index：为行业和分位数的MultiIndex
- columns：调仓周期

#### 最高分位数因子收益减最低分位数因子收益每日均值

```
far.mean_return_spread_by_quantile
```

- 类型：pandas.DataFrame
- index：日期
- columns：调仓周期

#### 最高分位数因子收益减最低分位数因子收益每日标准差

```
far.mean_return_spread_std_by_quantile
```

- 类型：pandas.DataFrame
- index：日期
- columns：调仓周期

#### 信息比率

```
far.ic
```

- 类型：pandas.DataFrame
- index：日期
- columns：调仓周期

#### 分行业信息比率

```
far.ic_by_group
```

- 类型：pandas.DataFrame
- index：行业
- columns：调仓周期

#### 月度信息比率

```
far.ic_monthly
```

- 类型：pandas.DataFrame
- index：月度
- columns：调仓周期表

#### 换手率

```
far.quantile_turnover
```

- 键：调仓周期
- 值: pandas.DataFrame 换手率
    - index：日期
    - columns：分位数分组

#### 计算按分位数分组加权因子收益和标准差

```
mean,std = far.calc_mean_return_by_quantile(by_date=True, by_group=False, demeaned=False, group_adjust=False)
```

**参数：**

- by_date：
    - True：在分位数分组下，得到每天的因子收益和标准差
    - False：按分位数分组加权因子收益和标准差
- by_group:
    - True: 在分位数分组下，得到每个行业的因子收益和标准差
    - False：按分位数分组加权因子收益和标准差
- demeaned:
    - True：对每天的因子收益和标准差去均值
    - False：按分位数分组加权因子收益和标准差
- group_adjust:
    - True：每天的因子收益和标准差按行业去均值
    - False：按分位数分组加权因子收益和标准差

#### 计算按因子值加权多空组合每日收益

```
far.calc_factor_returns(demeaned=True, group_adjust=False)
```

**参数：**

- demeaned:
    - True:对每天的因子收益去均值
    - False：每日因子收益
- group_adjust:
    - True：对每天的因子收益去均值按行业去均值
    - False：计算按因子值加权多空组合每日收益

#### 计算两个分位数相减的因子收益和标准差

```
mean, std = far.compute_mean_returns_spread (by_date=False, by_group=False, demeaned=False, group_adjust=False, upper_quant=1, lower_quant=8)
```

**参数：**

- by_date：
    - True：每天的两个分位数相减的因子收益和标准差
    - False：两个分位数相减的因子收益和标准差
- by_group:
    - True: 每个行业的两个分位数相减的因子收益和标准差
    - False：两个分位数相减的因子收益和标准差
- demeaned:
    - True：两个分位数相减去均值后的每天的因子收益和标准差
    - False：两个分位数相减的因子收益和标准差
- group_adjust:
    - True：两个分位数相减按行业去均值后的因子收益和标准差
    - False：两个分位数相减的因子收益和标准差
- upper_quant：用upper_quant选择的分位数减去lower_quant选择的分位数，只能在已有的范围内选择
- lower_quant：用upper_quant选择的分位数减去lower_quant选择的分位数，只能在已有的范围内选择

#### 计算因子的 alpha 和 beta

```
far.calc_factor_alpha_beta(demeaned=True, group_adjust=False)
```

**参数：**

- demeaned:
    - True:去均值后的因子收益计算得到的alpha和beta
    - False：计算因子的 alpha 和 beta
- group_adjust:
    - True：按行业去均值后的因子收益计算得到的alpha和beta
    - False：计算因子的 alpha 和 beta

#### 计算每日因子信息比率（IC值）

```
far.calc_factor_information_coefficient(group_adjust=False, by_group=False, method='rank')
```

**参数：**

- group_adjust:
    - True：按行业去均值后的因子收益计算得到的IC值
    - False：IC值
- by_group:
    - True：得到每日因子信息比率的各行业IC值
    - False：IC值
- method：
    - 'rank'：用秩相关系数计算IC值
    - 'normal'：用普通相关系数计算IC值

#### 计算因子信息比率均值（IC值均值）

```
far.calc_mean_information_coefficient(group_adjust=False, by_group=False, by_time=None, method='rank')
```

**参数：**

- group_adjust:
    - True：按行业去均值后的因子收益计算得到的IC值均值
    - False：IC值均值
- by_group:
    - True：各行业IC值均值
    - False：IC值均值
- by_time：
    - 'Y'：按年求均值
    - 'M'：按月求均值
    - None：对所有日期求均值
- method：
    - 'rank'：用秩相关系数计算IC值
    - 'normal'：用普通相关系数计算IC值

#### 按照当天的分位数算分位数未来和过去的收益均值和标准差

```
far.calc_average_cumulative_return_by_quantile(periods_before=5, periods_after=15, demeaned=False, group_adjust=False)
```

**参数：**

- periods_before：计算过去的天数
- periods_after：计算未来的天数
- demeaned：对每日因子收益去均值后计算未来和过去的收益均值和标准差
- group_adjust：按行业对因子收益去均值后计算未来和过去的收益均值和标准差

#### 计算指定调仓周期的各分位数每日累积收益

```
far.calc_cumulative_return_by_quantile(period=5)
```

**参数：**

- period：指定调仓周期

#### 计算指定调仓周期的按因子值加权多空组合每日累积收益

```
far.calc_cumulative_returns(period=5, demeaned=False, group_adjust=False)
```

**参数：**

- period：指定调仓周期
- demeaned:
    - True：对每日因子收益去均值后按因子值加权多空组合每日累积收益
    - False：按因子值加权多空组合每日累积收益
- group_adjust:
    - True：按行业对因子收益去均值后按因子值加权多空组合每日累积收益
    - False：按因子值加权多空组合每日累积收益

#### 计算指定调仓周期和前面定义好的加权方式计算多空组合每日累计收益

```
far.calc_top_down_cumulative_returns(period=5, demeaned=False, group_adjust=False)
```

**参数：**

- period：指定调仓周期
- demeaned:
    - True：对每日因子收益去均值后按因子值加权多空组合每日累积收益
    - False：按因子值加权多空组合每日累积收益
- group_adjust:
    - True：按行业对因子收益去均值后按因子值加权多空组合每日累积收益
    - False：按因子值加权多空组合每日累积收益

#### 根据调仓周期确定滞后期的每天计算因子自相关性

```
far.calc_autocorrelation(rank=True)
```

**参数：**

- rank：
    - True：秩相关系数
    - False：普通相关系数

#### 滞后n天因子值自相关性

```
far.calc_autocorrelation_n_days_lag(n=9,rank=True)
```

**参数：**

- n：滞后n天到1天的因子值自相关性
- rank：
    - True：秩相关系数
    - False：普通相关系数

#### 各分位数换手率n天的移动平均

```
far.calc_quantile_turnover_mean_n_days_lag(n=10)
```

**参数：**

- n:滞后n天到1天的换手率

#### 滞后 0 - n 天因子收益信息比率(IC)的移动平均

```
far.calc_ic_mean_n_days_lag(n=10,group_adjust=False,by_group=False,method=None)
```

**参数：**

- n：滞后0-n天因子收益的信息比率(IC)的移动平均
- group_adjust:
    - True：按行业对因子收益去均值后滞后0-n天因子收益的信息比率(IC)的移动平均
    - False：滞后 0 - n 天因子收益信息比率(IC)的移动平均
- by_group：
    - True：滞后 0 - n 天各行业的因子收益信息比率(IC)的移动平均
    - False：滞后 0 - n 天因子收益信息比率(IC)的移动平均
- method：
    - 'rank'：用秩相关系数计算IC值
    - 'normal'：用普通相关系数计算IC值



## 获取聚宽因子库数据的方法

1. [聚宽因子库](https://www.joinquant.com/help/api/help?name=factor_values)包含数百个质量、情绪、风险等其他类目的因子

2. 连接jqdatasdk获取数据包，数据接口需调用聚宽 [`jqdatasdk`](https://github.com/JoinQuant/jqdatasdk/blob/master/README.md) 接口获取金融数据([试用注册地址](http://t.cn/EINDOxE))

    ```python
    # 获取因子数据：以5日平均换手率为例，该数据可以直接用于因子分析
    # 具体使用方法可以参照jqdatasdk的API文档
    import jqdatasdk
    jqdatasdk.auth('username', 'password')
    # 获取聚宽因子库中的VOL5数据
    factor_data=jqdatasdk.get_factor_values(
        securities=jqdatasdk.get_index_stocks('000300.XSHG'),
        factors=['VOL5'],
        start_date='2018-01-01',
        end_date='2018-12-31')['VOL5']
    ```



## 将自有因子值转换成 DataFrame 格式的数据

- index 为日期，格式为 pandas 日期通用的 DatetimeIndex

- columns 为股票代码，格式要求符合聚宽的代码定义规则（如：平安银行的股票代码为 000001.XSHE）

    - 如果是深交所上市的股票，在股票代码后面需要加入.XSHE
    - 如果是上交所上市的股票，在股票代码后面需要加入.XSHG

- 将 pandas.DataFrame 转换成满足格式要求数据格式

    首先要保证 index 为 `DatetimeIndex` 格式

    一般是通过 pandas 提供的 [`pandas.to_datetime`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html) 函数进行转换, 在转换前应确保 index 中的值都为合理的日期格式, 如 `'2018-01-01'` / `'20180101'`, 之后再调用 `pandas.to_datetime` 进行转换

    另外应确保 index 的日期是按照从小到大的顺序排列的, 可以通过 [`sort_index`](https://pandas.pydata.org/pandas-docs/version/0.23.3/generated/pandas.DataFrame.sort_index.html) 进行排序

    最后请检查 columns 中的股票代码是否都满足聚宽的代码定义

    ```python
    import pandas as pd

    sample_data = pd.DataFrame(
        [[0.84, 0.43, 2.33, 0.86, 0.96],
         [1.06, 0.51, 2.60, 0.90, 1.09],
         [1.12, 0.54, 2.68, 0.94, 1.12],
         [1.07, 0.64, 2.65, 1.33, 1.15],
         [1.21, 0.73, 2.97, 1.65, 1.19]],
        index=['2018-01-02', '2018-01-03', '2018-01-04', '2018-01-05', '2018-01-08'],
        columns=['000001.XSHE', '000002.XSHE', '000063.XSHE', '000069.XSHE', '000100.XSHE']
    )

    print(sample_data)

    factor_data = sample_data.copy()
    # 将 index 转换为 DatetimeIndex
    factor_data.index = pd.to_datetime(factor_data.index)
    # 将 DataFrame 按照日期顺序排列
    factor_data = factor_data.sort_index()
    # 检查 columns 是否满足聚宽股票代码格式
    if not sample_data.columns.astype(str).str.match('\d{6}\.XSH[EG]').all():
        print("有不满足聚宽股票代码格式的股票")
        print(sample_data.columns[~sample_data.columns.astype(str).str.match('\d{6}\.XSH[EG]')])

    print(factor_data)
    ```

- 将键为日期, 值为各股票因子值的 `Series` 的 `dict` 转换成 `pandas.DataFrame`

    可以直接利用 `pandas.DataFrame` 生成

    ```python
    sample_data = \
    {'2018-01-02': pd.Seris([0.84, 0.43, 2.33, 0.86, 0.96],
                            index=['000001.XSHE', '000002.XSHE', '000063.XSHE', '000069.XSHE', '000100.XSHE']),
     '2018-01-03': pd.Seris([1.06, 0.51, 2.60, 0.90, 1.09],
                            index=['000001.XSHE', '000002.XSHE', '000063.XSHE', '000069.XSHE', '000100.XSHE']),
     '2018-01-04': pd.Seris([1.12, 0.54, 2.68, 0.94, 1.12],
                            index=['000001.XSHE', '000002.XSHE', '000063.XSHE', '000069.XSHE', '000100.XSHE']),
     '2018-01-05': pd.Seris([1.07, 0.64, 2.65, 1.33, 1.15],
                            index=['000001.XSHE', '000002.XSHE', '000063.XSHE', '000069.XSHE', '000100.XSHE']),
     '2018-01-08': pd.Seris([1.21, 0.73, 2.97, 1.65, 1.19],
                            index=['000001.XSHE', '000002.XSHE', '000063.XSHE', '000069.XSHE', '000100.XSHE'])}

    import pandas as pd
    # 直接调用 pd.DataFrame 将 dict 转换为 DataFrame
    factor_data = pd.DataFrame(data).T

    print(factor_data)

    # 之后请按照 DataFrame 的方法转换成满足格式要求数据格式
    ```

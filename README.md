

# jqfactor_analyzer

**聚宽单因子分析工具开源版**

---

聚宽单因子分析工具开源版是提供给用户进行因子分析的工具，提供了包括计算因子IC值，因子收益，因子换手率等各种详细指标，用户可以按照自己的需求查看因子详情。

欢迎加入jqfactor_analyzer交流群，QQ 群聊号码：779882614


## **安装**

```bash
pip install jqfactor_analyzer
```



## **升级**

```bash
pip install -U jqfactor_analyzer
```



## **具体使用方法**

[analyze_factor](docs/API文档.md): 因子分析函数



## **使用示例**

* ### 示例：5日平均换手率因子分析

```python
# 载入函数库
import pandas as pd
import jqfactor_analyzer as ja

# 获取 jqdatasdk 授权，输入用户名、密码，申请地址：http://t.cn/EINDOxE
# 聚宽官网及金融终端，使用方法参见：http://t.cn/EINcS4j
import jqdatasdk
jqdatasdk.auth('username', 'password')

# 获取5日平均换手率因子2018-01-01到2018-12-31之间的数据（示例用从库中直接调取）
# 聚宽因子库数据获取方法在下方
from jqfactor_analyzer.sample import VOL5
factor_data = VOL5

# 对因子进行分析
far = ja.analyze_factor(
    factor_data,  # factor_data 为因子值的 pandas.DataFrame
    quantiles=10,
    periods=(1, 10),
    industry='jq_l1',
    weight_method='avg',
    max_loss=0.1
)

# 获取整理后的因子的IC值
far.ic
```

结果展示：

![1](http://img0.ph.126.net/yJ8JpnMULEAqE4hzaGzMcg==/861876378788739324.png)

```python
# 生成统计图表
far.create_full_tear_sheet(
    demeaned=False, group_adjust=False, by_group=False,
    turnover_periods=None, avgretplot=(5, 15), std_bar=False
)
```

结果展示：

![2](https://image.joinquant.com/88e0de9b43424e3b7b1ab1fe48514625)

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

* index 为日期，格式为 pandas 日期通用的 DatetimeIndex

* columns 为股票代码，格式要求符合聚宽的代码定义规则（如：平安银行的股票代码为 000001.XSHE）
  * 如果是深交所上市的股票，在股票代码后面需要加入.XSHE
  * 如果是上交所上市的股票，在股票代码后面需要加入.XSHG

* 将 pandas.DataFrame 转换成满足格式要求数据格式

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

* 将键为日期, 值为各股票因子值的 `Series` 的 `dict` 转换成 `pandas.DataFrame`

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

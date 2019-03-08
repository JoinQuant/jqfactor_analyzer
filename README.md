

# jqfactor_analyzer

**聚宽单因子分析工具开源版**

---

聚宽单因子分析工具开源版是提供给用户进行因子分析的工具，提供了包括计算因子IC值，因子收益，因子换手率等各种详细指标，用户可以按照自己的需求查看因子详情。



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

![2](https://image.joinquant.com/639f56bd353409e10d46f211e6a47023)

## **获取聚宽因子库数据的方法**

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

3. 自己算好因子值需要转换成 pandas.DataFrame 格式的数据

   * index 为日期，格式为 pandas 日期通用的 DatetimeIndex

   * columns 为股票代码，格式要求符合聚宽的代码定义规则（如：平安银行的股票代码为 000001.XSHE）
     * 如果是深交所上市的股票，在股票代码后面需要加入.XSHE
     * 如果是上交所上市的股票，在股票代码后面需要加入.XSHG

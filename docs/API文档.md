# **API文档**

## 一、因子缓存factor_cache模块

为了在本地进行分析时，为了提高数据获取的速度并避免反复从服务端获取数据，所以增加了本地数据缓存的方法。

注意缓存格式为pyarrow.feather格式，pyarrow库不同版本之间可能存在兼容问题，建议不要随意修改pyarrow库的版本，如果修改后产生大量缓存文件无法读取(提示已损坏)的情况，建议删除整个缓存目录后重新缓存。

### 1. 设置缓存目录

对于单因子分析和归因分析中使用到的市值/价格和风格因子等数据，默认会缓存到用户的主目录( `os.path.expanduser( '~/jqfactor_datacache/bundle')` )。 一般地，在 Unix 系统上可能是 `/home/username/jqfactor_datacache/bundle`，而在 Windows 系统上可能是 `C:\Users\username\jqfactor_datacache\bundle`。

您可以通过以下代码修改配置信息来设置为其他路径，设置过一次后后续都将沿用设置的这个路径，不用重复设置。

```python
from jqfactor_analyzer.factor_cache import set_cache_dir,get_cache_dir
set_cache_dir(my_path) #设置缓存目录为my_path
print(get_cache_dir()) #输出缓存目录
```

### 2. 缓存/检查缓存和读取已缓存数据

除过对单因子分析及归因分析依赖的数据进行缓存外，factor_cache还可以缓存自定义的因子组(仅限聚宽因子库中支持的因子)

```python
def save_factor_values_by_group(start_date,end_date,factor_names='prices',
				group_name=None,overwrite=False,cache_dir=None,show_progress=True):
    """将因子库数据按因子组储存到本地,根据factor_names因子列表(顺序无关)自动生成因子组的名称
    start_date : 开始时间
    end_date : 结束时间
    factor_names : 因子组所含因子的名称,除过因子库中支持的因子外，还支持指定为'prices'缓存价格数据
    group_name : 因子组名称，不指定时使用get_factor_folder自动生成因子组名(即缓存文件夹名)，如果指定则按照指定的名称生成文件夹名(使用get_factor_values_by_cache时,需要自行指定factor_path)
    overwrite  : 文件已存在时是否覆盖更新,默认为False即增量更新,文件已存在时跳过
    cache_dir : 缓存的路径，如果没有指定则使用配置信息中的路径,一般不用指定
    show_progress : 是否展示缓存进度,默认为True
    返回 : 因子组储存的路径 , 文件以天为单位储存为feather文件,每天一个feather文件,每月一个文件夹,columns为因子名称, index为当天在市的所有标的代码
    """
def get_factor_values_by_cache(date,codes=None,factor_names=None,group_name=None,
								factor_path=None):
    """从缓存的文件读取因子数据,文件不存在时返回空的dataframe
    date : 日期
    codes : 标的代码,默认为None获取当天在市的所有标的
    factor_names : 因子列表(顺序无关),当指定factor_path/group_name时失效
    group_name : 因子组名,如果缓存时指定了group_name,则获取时必须也指定group_name或factor_path
    factor_path : 可选参数,因子组的路径,一般不用指定
    返回:
    如果缓存文件存在，则返回当天的因子数据,index是标的代码,columns是因子名
    如果缓存文件不存在,则返回空的dataframe, 建议在使用get_factor_values_by_cache前,先运行save_factor_values_by_group检查时间区间内的缓存文件是否完整
    """
def get_factor_folder(factor_names,group_name=None):
    """获取因子组的文件夹名(文件夹位于get_cache_dir()获取的缓存目录下)
    factor_names : 因子储存时,如果未指定group_name,则根据因子列表(顺序无关)获取md5值生成因子组名(即储存的文件夹名)，使用此方法可以获取生成的文件夹名称
    group_name : 如果储存时指定了因子组名,则直接返回此因子组名
    """

```

**示例**

```python
from jqfactor_analyzer.factor_cache import save_factor_values_by_group,get_factor_values_by_cache,get_factor_folder,get_cache_dir
# import jqdatasdk as jq
# jq.auth("账号",'密码') #登陆jqdatasdk来从服务端缓存数据

all_factors = jq.get_all_factors()
factor_names = all_factors[all_factors.category=='growth'].factor.tolist()  #将聚宽因子库中的成长类因子作为一组因子
group_name = 'growth_factors' #因子组名定义为'growth_factors'
start_date = '2021-01-01'
end_date = '2021-06-01'
# 检查/缓存因子数据
factor_path = save_factor_values_by_group(start_date,end_date,factor_names=factor_names,group_name=group_name,overwrite=False,show_progress=True)
# factor_path = os.path.join(get_cache_dir(), get_factor_folder(factor_names,group_name=group_name)  #等同于save_factor_values_by_group返回的路径

# 循环获取缓存的因子数据,并拼接
trade_days = jq.get_trade_days(start_date,end_date)
factor_values = {}
for date in trade_days:
    factor_values[date] = get_factor_values_by_cache(date,codes=None,factor_names=factor_names,group_name=group_name, factor_path=factor_path)#这里实际只需要指定group_name,factor_names参数的其中一个,缓存时指定了group_name时,factor_names不生效
factor_values = pd.concat(factor_values)
```

## 二、归因分析模块

```python
from jqfactor_analyzer import AttributionAnalysis
AttributionAnalysis(weights,daily_return,style_type='style_pro',industry ='sw_l1',use_cn=True,show_data_progress=True)
```

**参数 :**

- `weights`:持仓权重信息，index是日期，columns是标的代码， value对应的是组合当天的仓位占比(单日仓位占比总和不为1时，剩余部分认为是当天的现金)
-  `daily_return`:Series,index是日期，values为当天组合的收益率
-  `style_type`:归因分析所使用的风格因子类型，可选'style'和'style_pro'中的一个
-  `industry`:归因分析所使用的行业分类，可选'sw_l1'和'jq_l1'中的一个
-  `use_cn`:绘图时是否使用中文
-  `show_data_progress`:是否展示数据获取进度(使用本地缓存,第一次运行时速度较慢,后续对于本地不存在的数据将增量缓存)

**示例**

```python
import pandas as pd
# position_weights.csv 是一个储存了组合权重信息的csv文件,index是日期,columns是股票代码
# position_daily_return.csv 是一个储存了组合日收益率的csv文件,index是日期,daily_return列是日收益
weights = pd.read_csv("position_weights.csv",index_col=0)
returns = pd.read_csv("position_daily_return.csv",index_col=0)['daily_return']

An =  AttributionAnalysis(weights , returns ,style_type='style_pro',industry ='sw_l1', show_data_progress=True )
```



### 1. 属性

- `style_exposure` : 组合的风格暴露
- `industry_exposure` : 组合的行业暴露
- `exposure_portfolio` : 组合的风格+行业及country暴露
- `attr_daily_returns` : 组合的\风格+行业及country日度归因收益率
- `attr_returns` : 组合的日度风格+行业及country累积归因收益率

### 2. 方法

#### (1) 获取组合相对于指数的暴露

```python
get_exposure2bench(index_symbol)
```

**参数 :**

- `index_symbol` : 基准指数, 可选`['000300.XSHG','000905.XSHG','000906.XSHG','000852.XSHG','932000.CSI','000985.XSHG']`中的一个

**返回 :**

- 一个dataframe,index为日期,columns为风格因子+行业因子+county , 其中country为股票总持仓占比

#### (2) 获取组合相对于指数的日度归因收益率

```python
get_attr_daily_returns2bench(index_symbol)
```

假设组合相对于指数的收益由以下部分构成 : 风格+行业暴露收益(common_return ) , 现金闲置收益(cash) ,策略本身的超额收益(specific_return)
**参数 :**

- `index_symbol` : 基准指数, 可选`['000300.XSHG','000905.XSHG','000906.XSHG','000852.XSHG','932000.CSI','000985.XSHG']`中的一个

**返回 :**

- 一个dataframe,index为日期,columns为`风格因子+行业因子+cash+common_return,specific_return,total_return`

  其中:
  cash是假设现金收益(0)相对指数带来的收益率
  common_return 为风格+行业总收益率
  specific_return 为特意收益率
  total_return 为组合相对于指数的总收益

#### (3) 获取相对于指数的累积归因收益率

```python
get_attr_returns2bench(index_symbol)
```

假设组合相对于指数的收益由以下部分构成 : 风格+行业暴露收益(common_return ) , 现金闲置收益(cash) ,策略本身的超额收益(specific_return)

**参数 :**

 `index_symbol` : 基准指数, 可选`['000300.XSHG','000905.XSHG','000906.XSHG','000852.XSHG','932000.CSI','000985.XSHG']`中的一个

**返回 :**

- 一个dataframe,index为日期,columns为`风格因子+行业因子+cash+common_return,specific_return,total_return`

  其中:
  cash是假设现金收益(0)相对指数带来的收益率
  common_return 为风格+行业总收益率
  specific_return 为特异收益率
  total_return 为组合相对于指数的总收益(减法超额)

### 3. 绘图方法

#### (1) 绘制风格暴露时序图

```python
plot_exposure(factors='style',index_symbol=None,figsize=(15,7))
```

绘制风格暴露时序

**参数**

- factors : 绘制的暴露类型 , 可选 'style'(所有风格因子) , 'industry'(所有行业因子),也可以传递一个list,list为exposure_portfolio中columns的一个或者多个
- index_symbol : 基准指数代码,指定时绘制相对于指数的暴露 , 默认None为组合本身的暴露
- figsize : 画布大小

#### (2) 绘制归因分析收益时序图

```python
plot_returns(factors='style',index_symbol=None,figsize=(15,7))
```

绘制归因分析收益时序

**参数**

- factors : 绘制的暴露类型 , 可选 'style'(所有风格因子) , 'industry'(所有行业因子),也可以传递一个list,list为exposure_portfolio中columns的一个或者多个
  同时也支持指定['common_return'(风格总收益),'specific_return'(特异收益),'total_return'(总收益)', 'country'(国家因子收益,当指定index_symbol时会用现金相对于指数的收益替代)]
- index_symbol : 基准指数代码,指定时绘制相对于指数的暴露 , 默认None为组合本身的暴露
- figsize : 画布大小

#### (3) 绘制暴露与收益对照图

```python
plot_exposure_and_returns(factors='style',index_symbol=None,show_factor_perf=False,figsize=(12,6))
```

将因子暴露与收益同时绘制在多个子图上

**参数**

-  factors : 绘制的暴露类型 , 可选 'style'(所有风格因子) , 'industry'(所有行业因子),也可以传递一个list,list为exposure_portfolio中columns的一个或者多个
  当指定index_symbol时,country会用现金相对于指数的收益替代)
- index_symbol : 基准指数代码,指定时绘制相对于指数的暴露及收益 , 默认None为组合本身的暴露和收益
- show_factor_perf : 是否同时绘制因子表现
- figsize : 画布大小,这里第一个参数是画布的宽度, 第二个参数为单个子图的高度

#### (4) 关闭中文图例显示

```python
plot_disable_chinese_label()
```

 画图时默认会从系统中查找中文字体显示以中文图例
 如果找不到中文字体则默认使用英文图例
 当找到中文字体但中文显示乱码时, 可调用此 API 关闭中文图例显示而使用英文



## 三、单因子分析模块

```python
from jqfactor_analyzer import analyze_factor
analyze_factor(factor, industry='jq_l1', quantiles=5, periods=(1, 5, 10), weight_method='avg', max_loss=0.25, allow_cache=True, show_data_progress=True )
```

单因子分析函数



**参数**

* `factor`: 因子值，

  pandas.DataFrame格式的数据

  - index为日期，格式为pandas日期通用的DatetimeIndex，转换方法见[将自有因子值转换成 DataFrame 格式的数据](#将自有因子值转换成-dataframe-格式的数据)
  - columns为股票代码，格式要求符合聚宽的代码定义规则（如：平安银行的股票代码为000001.XSHE）
      - 如果是深交所上市的股票，在股票代码后面需要加入.XSHE
      - 如果是上交所上市的股票，在股票代码后面需要加入.XSHG

  或 pd.Series格式的数据
  - index为日期和股票代码组成的MultiIndex

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

* `allow_cache` : 是否允许对价格,市值等信息进行本地缓存(按天缓存,初次运行可能比较慢,但后续重新获取对应区间的数据将非常快,且分析时仅消耗较小的jqdatasdk流量)

* show_data_progress: 是否展示数据获取的进度信息



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







### 1. 绘制结果

#### 展示全部分析

```
far.create_full_tear_sheet(demeaned=False, group_adjust=False, by_group=False,
turnover_periods=None, avgretplot=(5, 15), std_bar=False)
```

**参数:**

- demeaned:
    - True: 使用超额收益计算 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
    - False: 不使用超额收益
- group_adjust:
    - True: 使用行业中性化后的收益计算 (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
    - False: 不使用行业中性化后的收益
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
    - True: 使用超额收益计算 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
    - False: 不使用超额收益
- group_adjust:
    - True: 使用行业中性化后的收益计算 (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
    - False: 不使用行业中性化后的收益
- by_group:
    - True: 画各行业的各分位数平均收益图
    - False: 不画各行业的各分位数平均收益图

#### 因子 IC 分析

```
far.create_information_tear_sheet(group_adjust=False, by_group=False)

```

**参数:**

- group_adjust:
    - True: 使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
    - False: 不使用行业中性收益
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
    - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
    - False: 不使用超额收益
- group_adjust:
    - True: 使用行业中性化后的收益计算累积收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
    - False: 不使用行业中性化后的收益
- std_bar:
    - True: 显示标准差
    - False: 不显示标准差

#### 打印因子收益表

```
far.plot_returns_table(demeaned=False, group_adjust=False)

```

**参数：**

- demeaned:
    - True：使用超额收益计算 (基准收益被认为是每日所有股票收益按照weight列中权重的加权的均值)
    - False：不使用超额收益
- group_adjust:
    - True：使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
    - False：不使用行业中性收益

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
    - True：使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
    - False：不使用行业中性收益
- method：
    - 'rank'：用秩相关系数计算IC值
    - 'normal': 用相关系数计算IC值

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
    - True：使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
    - False：不使用行业中性收益
- method：
    - 'rank'：用秩相关系数计算IC值
    - 'normal': 用相关系数计算IC值

#### 画信息比率分布直方图

```
far.plot_ic_hist(group_adjust=False, method='rank')

```

**参数：**

- group_adjust:
    - True：使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
    - False：不使用行业中性收益
- method：
    - 'rank'：用秩相关系数计算IC值
    - 'normal': 用相关系数计算IC值

#### 画信息比率 qq 图

```
far.plot_ic_qq(group_adjust=False, method='rank', theoretical_dist='norm')

```

**参数：**

- group_adjust:
    - True：使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
    - False：不使用行业中性收益
- method：
    - 'rank'：用秩相关系数计算IC值
    - 'normal': 用相关系数计算IC值
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
    - True：使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
    - False：不使用超额收益
- group_adjust:
    - True：使用行业中性化后的收益计算累积收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
    - False：不使用行业中性化后的收益

#### 画最高分位减最低分位收益图

```
far.plot_mean_quantile_returns_spread_time_series(demeaned=False, group_adjust=False, bandwidth=1)

```

**参数：**

- demeaned:
    - True：使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
    - False：不使用超额收益
- group_adjust:
    - True：使用行业中性化后的收益计算累积收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
    - False：不使用行业中性化后的收益
- bandwidth：n，加减n倍当日标准差

#### 画按行业分组信息比率(IC)图

```
far.plot_ic_by_group(group_adjust=False, method='rank')

```

**参数：**

- group_adjust:
    - True：使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
    - False：不使用行业中性收益
- method：
    - 'rank'：用秩相关系数计算IC值
    - 'normal': 用相关系数计算IC值

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
    - True：使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
    - False：不使用行业中性收益

#### 画按因子值加权多空组合每日累积收益图

```
far.plot_cumulative_returns(period=1, demeaned=False, group_adjust=False)

```

**参数：**

- periods：调仓周期
- demeaned:
    - True：对因子值加权组合每日收益的权重去均值 (每日权重 = 每日权重 - 每日权重的均值)，使组合转换为cash-neutral多空组合
    - False：不对权重去均值
- group_adjust:
    - True：对权重分行业去均值 (每日权重 = 每日权重 - 每日各行业权重的均值)，使组合转换为 industry-neutral 多空组合
    - False：不对权重分行业去均值

#### 画做多最大分位数做空最小分位数组合每日累积收益图

```
far.plot_top_down_cumulative_returns(period=1, demeaned=False, group_adjust=False)

```

**参数：**

- periods：指定调仓周期
- demeaned:
    - True：使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
    - False：不使用超额收益
- group_adjust:
    - True：使用行业中性化后的收益计算累积收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
    - False：不使用行业中性化后的收益

#### 画各分位数每日累积收益图

```
far.plot_cumulative_returns_by_quantile(period=(1, 3, 9), demeaned=False, group_adjust=False)

```

**参数：**

- periods：调仓周期
- demeaned:
    - True：使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
    - False：不使用超额收益
- group_adjust:
    - True：使用行业中性化后的收益计算累积收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
    - False：不使用行业中性化后的收益

#### 因子预测能力平均累计收益图

```
far.plot_quantile_average_cumulative_return(periods_before=5, periods_after=10, by_quantile=False, std_bar=False, demeaned=False, group_adjust=False)

```

**参数：**

- periods_before: 计算过去的天数
- periods_after: 计算未来的天数
- by_quantile：
    - True：各分位数分别显示因子预测能力平均累计收益图
    - False：不用各分位数分别显示因子预测能力平均累计收益图
- std_bar：
    - True：显示标准差
    - False：不显示标准差
- demeaned:
    - True: 使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
    - False: 不使用超额收益
- group_adjust:
    - True: 使用行业中性化后的收益计算累积收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
    - False: 不使用行业中性化后的收益

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



### 2. 属性列表

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
mean, std = far.calc_mean_return_by_quantile(by_date=True, by_group=False, demeaned=False, group_adjust=False)
```

**参数：**

- by_date：
    - True：按天计算收益
    - False：不按天计算收益
- by_group:
    - True: 按行业计算收益
    - False：不按行业计算收益
- demeaned:
    - True：使用超额收益计算各分位数收益，超额收益=收益-基准收益 (基准收益被认为是每日所有股票收益按照weight列中权重的加权的均值)
    - False：不使用超额收益
- group_adjust:
    - True：使用行业中性收益计算各分位数收益，行业中性收益=收益-行业收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重的加权的均值)
    - False：不使用行业中性收益

#### 计算按因子值加权多空组合每日收益

```
far.calc_factor_returns(demeaned=True, group_adjust=False)
```

权重 = 每日因子值 / 每日因子值的绝对值的和
正的权重代表买入, 负的权重代表卖出

**参数：**

- demeaned:
    - True: 对权重去均值 (每日权重 = 每日权重 - 每日权重的均值), 使组合转换为 cash-neutral 多空组合
    - False：不对权重去均值
- group_adjust:
    - True：对权重分行业去均值 (每日权重 = 每日权重 - 每日各行业权重的均值)，使组合转换为 industry-neutral 多空组合
    - False：不对权重分行业去均值

#### 计算两个分位数相减的因子收益和标准差

```
mean, std = far.compute_mean_returns_spread(upper_quant=None, lower_quant=None, by_date=False, by_group=False, demeaned=False, group_adjust=False)
```

**参数：**

- upper_quant：用upper_quant选择的分位数减去lower_quant选择的分位数，只能在已有的范围内选择
- lower_quant：用upper_quant选择的分位数减去lower_quant选择的分位数，只能在已有的范围内选择
- by_date：
    - True：按天计算两个分位数相减的因子收益和标准差
    - False：不按天计算两个分位数相减的因子收益和标准差
- by_group:
    - True: 分行业计算两个分位数相减的因子收益和标准差
    - False：不分行业计算两个分位数相减的因子收益和标准差
- demeaned:
    - True：使用超额收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
    - False：不使用超额收益
- group_adjust:
    - True：使用行业中性收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
    - False：不使用行业中性收益


#### 计算因子的 alpha 和 beta

```
far.calc_factor_alpha_beta(demeaned=True, group_adjust=False)
```

因子值加权组合每日收益 = beta * 市场组合每日收益 + alpha

因子值加权组合每日收益计算方法见 calc_factor_returns 函数

市场组合每日收益是每日所有股票收益按照weight列中权重加权的均值

结果中的 alpha 是年化 alpha

**参数：**

- demeaned:
    - True: 对因子值加权组合每日收益的权重去均值 (每日权重 = 每日权重 - 每日权重的均值)，使组合转换为cash-neutral多空组合
    - False：不对权重去均值
- group_adjust:
    - True：对权重分行业去均值 (每日权重 = 每日权重 - 每日各行业权重的均值)，使组合转换为 industry-neutral 多空组合
    - False：不对权重分行业去均值

#### 计算每日因子信息比率（IC值）

```
far.calc_factor_information_coefficient(group_adjust=False, by_group=False, method='rank')
```

**参数：**

- group_adjust:
    - True：使用行业中性收益计算 IC (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
    - False：不使用行业中性收益
- by_group:
    - True：分行业计算 IC
    - False：不分行业计算 IC
- method：
    - 'rank'：用秩相关系数计算IC值
    - 'normal'：用普通相关系数计算IC值

#### 计算因子信息比率均值（IC值均值）

```
far.calc_mean_information_coefficient(group_adjust=False, by_group=False, by_time=None, method='rank')
```

**参数：**

- group_adjust:
    - True：使用行业中性收益计算 IC (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
    - False：不使用行业中性收益
- by_group:
    - True：分行业计算 IC
    - False：不分行业计算 IC
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
- demeaned：
    - True：使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
    - False：不使用超额收益
- group_adjust：
    - True：使用行业中性化后的收益计算累积收益
    - False：不使用行业中性化后的收益

#### 计算指定调仓周期的各分位数每日累积收益

```
far.calc_cumulative_return_by_quantile(period=None, demeaned=False, group_adjust=False)
```

**参数：**

- period：指定调仓周期
- demeaned：
    - True：使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
    - False：不使用超额收益
- group_adjust：
    - True：使用行业中性化后的收益计算累积收益
    - False：不使用行业中性化后的收益

#### 计算指定调仓周期的按因子值加权多空组合每日累积收益

```
far.calc_cumulative_returns(period=5, demeaned=False, group_adjust=False)
```

当 period > 1 时，组合的累积收益计算方法为：

组合每日收益 = （从第0天开始每period天一调仓的组合每日收益 + 从第1天开始每period天一调仓的组合每日收益 + ... + 从第period-1天开始每period天一调仓的组合每日收益) / period

组合累积收益 = 组合每日收益的累积

**参数：**

- period：指定调仓周期
- demeaned:
    - True：对权重去均值 (每日权重 = 每日权重 - 每日权重的均值)，使组合转换为 cash-neutral 多空组合
    - False：不对权重去均值
- group_adjust:
    - True：对权重分行业去均值 (每日权重 = 每日权重 - 每日各行业权重的均值)，使组合转换为 industry-neutral 多空组合
    - False：不对权重分行业去均值

#### 计算指定调仓周期和前面定义好的加权方式计算多空组合每日累计收益

```
far.calc_top_down_cumulative_returns(period=5, demeaned=False, group_adjust=False)
```

**参数：**

- period：指定调仓周期
- demeaned:
    - True：使用超额收益计算累积收益 (基准收益被认为是每日所有股票收益按照weight列中权重加权的均值)
    - False：不使用超额收益
- group_adjust:
    - True：使用行业中性化后的收益计算累积收益 (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
    - False：不使用行业中性化后的收益

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

#### 各分位数滞后1天到n天的换手率均值

```
far.calc_quantile_turnover_mean_n_days_lag(n=10)
```

**参数：**

- n：滞后 1 天到 n 天的换手率

#### 滞后 0 - n 天因子收益信息比率(IC)的移动平均

```
far.calc_ic_mean_n_days_lag(n=10,group_adjust=False,by_group=False,method=None)
```

**参数：**

- n：滞后0-n天因子收益的信息比率(IC)的移动平均
- group_adjust:
    - True：使用行业中性收益计算 IC (行业收益被认为是每日各个行业股票收益按照weight列中权重加权的均值)
    - False：不使用行业中性收益
- by_group：
    - True：分行业计算 IC
    - False：不分行业计算 IC
- method：
    - 'rank'：用秩相关系数计算IC值
    - 'normal'：用普通相关系数计算IC值



### 3. 获取聚宽因子库数据的方法

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



### 4. 将自有因子值转换成 DataFrame 格式的数据

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

## 四、数据处理函数

#### 1.  中性化

```python
from jqfactor_analyzer import neutralize
neutralize(data, how=None, date=None, axis=1, fillna=None, add_constant=False)
```

**参数 :**

- data: pd.Series/pd.DataFrame, 待中性化的序列, 序列的 index/columns 为股票的 code
- how: str list. 中性化使用的因子名称列表.
  默认为 ['jq_l1', 'market_cap'], 支持的中性化方法有:
  (1) 行业: sw_l1, sw_l2, sw_l3, jq_l1, jq_l2
  (2) mktcap(总市值), ln_mktcap(对数总市值), cmktcap(流通市值), ln_cmktcap(对数流通市值)
  (3)自定义的中性化数据: 支持同时传入额外的 Series 或者 DataFrame 用来进行中性化, index 必须是标的代码
  数列表。
- date: 日期, 将用 date 这天的相关变量数据对 series 进行中性化 (注意依赖数据的实际可用时间, 如市值数据当天盘中是无法获取到的)
- axis: 默认为 1. 仅在 data 为 pd.DataFrame 时生效. 表示沿哪个方向做中性化, 0 为对每列做中性化, 1 为对每行做中性化
- fillna: 缺失值填充方式, 默认为None, 表示不填充. 支持的值:
          'jq_l1': 聚宽一级行业
          'jq_l2': 聚宽二级行业
          'sw_l1': 申万一级行业
          'sw_l2': 申万二级行业
          'sw_l3': 申万三级行业 表示使用某行业分类的均值进行填充.
-  add_constant: 中性化时是否添加常数项, 默认为 False

**返回 :**

- 中性化后的因子数据



#### 2.  去极值

```python
from jqfactor_analyzer import winsorize
winsorize(data, scale=None, range=None, qrange=None, inclusive=True, inf2nan=True, axis=1)
```

**参数 :**

- data: pd.Series/pd.DataFrame/np.array, 待缩尾的序列
- scale: 标准差倍数，与 range，qrange 三选一，不可同时使用。会将位于 [mu - scale * sigma, mu + scale * sigma] 边界之外的值替换为边界值
- range: 列表， 缩尾的上下边界。与 scale，qrange 三选一，不可同时使用。
- qrange: 列表，缩尾的上下分位数边界，值应在 0 到 1 之间，如 [0.05, 0.95]。与 scale，range 三选一，不可同时使用。
- inclusive: 是否将位于边界之外的值替换为边界值，默认为 True。如果为 True，则将边界之外的值替换为边界值，否则则替换为 np.nan
- inf2nan: 是否将 np.inf 和 -np.inf 替换成 np.nan，默认为 True如果为 True，在缩尾之前会先将 np.inf 和 -np.inf 替换成 np.nan，缩尾的时候不会考虑 np.nan，否则 inf 被认为是在上界之上，-inf 被认为在下界之下
- axis: 在 data 为 pd.DataFrame 时使用，沿哪个方向做标准化，默认为 1。 0 为对每列做缩尾，1 为对每行做缩尾。

**返回 :**

- 去极值处理之后的因子数据



#### 3.  中位数去极值

```python
from jqfactor_analyzer import winsorize_med
winsorize_med(data, scale=1, inclusive=True, inf2nan=True, axis=1)
```

**参数 :**

- data: pd.Series/pd.DataFrame/np.array, 待缩尾的序列
- scale: 倍数，默认为 1.0。会将位于 [med - scale * distance, med + scale * distance] 边界之外的值替换为边界值/np.nan
- inclusive bool 是否将位于边界之外的值替换为边界值，默认为 True。 如果为 True，则将边界之外的值替换为边界值，否则则替换为 np.nan
- inf2nan: 是否将 np.inf 和 -np.inf 替换成 np.nan，默认为 True。如果为 True，在缩尾之前会先将 np.inf 和 -np.inf 替换成 np.nan，缩尾的时候不会考虑 np.nan，否则 inf 被认为是在上界之上，-inf 被认为在下界之下
- axis: 在 data 为 pd.DataFrame 时使用，沿哪个方向做标准化，默认为 1。0 为对每列做缩尾，1 为对每行做缩尾

**返回 :**

- 中位数去极值之后的因子数据



#### 4.  标准化(z-score)

```python
from jqfactor_analyzer import standardlize
standardlize(data, inf2nan=True, axis=1)
```

**参数 :**

- data: pd.Series/pd.DataFrame/np.array, 待标准化的序列
- inf2nan: 是否将 np.inf 和 -np.inf 替换成 np.nan。默认为 True
- axis=1: 在 data 为 pd.DataFrame 时使用，如果 series 为 pd.DataFrame，沿哪个方向做标准化。0 为对每列做标准化，1 为对每行做标准化

**返回 :**

- 标准化后的因子数据

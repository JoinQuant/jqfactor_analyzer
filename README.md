# jqfactor_analyzer
jqfactor_analyzer 是提供给用户配合 jqdatasdk 进行归因分析，因子数据缓存及单因子分析的开源工具。

### 安装
```pip install jqfactor_analyzer```
### 升级
```pip install -U jqfactor_analyzer```

### 具体使用方法
**详细用法请查看[API文档](https://github.com/JoinQuant/jqfactor_analyzer/blob/master/docs/API%E6%96%87%E6%A1%A3.md)**


## 归因分析使用示例

### 风格模型的基本概念
归因分析旨在通过对历史投资组合的收益进行分解，明确指出各个收益来源对组合的业绩贡献，能够更好地理解组合的表现是否符合预期，以及是否存在某一风格/行业暴露过高的风险。

多因子风险模型的基础理论认为，股票的收益是由一些共同的因子 (风格，行业和国家因子) 来驱动的，不能被这些因子解释的部分被称为股票的 “特异收益”， 而每只股票的特异收益之间是互不相关的。

(1) 风格因子，即影响股票收益的风格因素，如市值、成长、杠杆等。

(2) 行业因子，不同行业在不同时期可能优于或者差于其他行业，同一行业内的股票往往涨跌具有较强的关联性。

(3) 国家因子，表示股票市场整体涨落对投资组合的收益影响，对于任意投资组合，若他们投资的都是同一市场则其承担的国家因子和收益是相同的。

(4) 特异收益，即无法被多因子风险模型解释的部分，也就是影响个股收益的特殊因素，如公司经营能力、决策等。

根据上述多因子风险模型，股票的收益可以表达为 :

$$
R_i = \underbrace{1 \cdot f_c} _{\text{国家因子收益}} + \underbrace{\sum _{j=1}^{S} f _j^{style} \cdot X _{ij}^{style}} _{\text{风格因子收益}} + \underbrace{\sum _{j=1}^{I} f _j^{industry} \cdot X _{ij}^{industry}} _{\text{行业因子收益}} + \underbrace{u _i} _{\text{个股特异收益}}
$$

此公式可简化为:

$$
R_i = \underbrace{\sum_{j=1}^{K} f_j \cdot X_{ij}}_{\text{第 j 个因子 (含国家，风格和行业，总数为 K) 获得的收益}} + \underbrace{u_i} _{\text{个股特异收益}}
$$

其中：
- $R_i$ 是第 $i$ 只股票的收益
- $f_c$ 是国家因子的回报率
- $S$ 和 $I$ 分别是风格和行业因子的数量
- $f_j^{style}$ 是第 $j$ 个风格因子的回报率, $f_j^{industry}$ 是第 $j$ 个行业因子的回报率
- $X_{ij}^{style}$ 是第 $i$ 只股票在第 $j$ 个风格因子上的暴露, $X_{ij}^{industry}$ 是第 $i$ 只股票在第 $j$ 个行业因子上的暴露，因子暴露又称因子载荷/因子值 (通过<span style="color:red;">`jqdatasdk.get_factor_values`</span>可获取风格因子暴露及行业暴露哑变量)
- $u_i$ 是残差项，表示无法通过模型解释的部分 (即特异收益率)

根据上述公式，对市场上的股票 (一般采用中证全指作为股票池) 使用对数市值加权在横截面上进行加权最小二乘回归，可得到 :
- $f_j$ : 风格/行业因子和国家因子的回报率 ， 通过 <span style="color:red;">`jqdatasdk.get_factor_style_returns`</span> 获取
- $u_i$ : 回归残差 (无法被模型解释的部分，即特异收益率)， 通过<span style="color:red;"> `jqdatasdk.get_factor_specific_returns` </span>获取


### 使用上述已提供的数据进行归因分析 :
现已知你的投资组合 P 由权重 $w_n$ 构成，则投资组合第 j 个因子的暴露可表示为 :

$$
X^P_j = \sum_{i=1}^{n} w_i X_{ij}
$$

- $X^P_j$ 可通过 <span style="color:red;">`jqfactor_analyzer.AttributionAnalysis().exposure_portfolio` </span>获取


投资组合在第 j 个因子上获取到的收益率可以表示为 :

$$
R^P_j = X^P_j \cdot f_j
$$

- $R^P_j$ 可通过 <span style="color:red;">`jqfactor_analyzer.AttributionAnalysis().attr_daily_return` </span>获取

所以投资组合的收益率也可以被表示为 :

$$
R_P = \sum_{j=1}^{k} R^p_j \cdot f_j + \sum_{i-1}^{n} w_i u_i
$$

即理论上 $\sum_n w_n u_n$  就是投资组合的特异收益 (alpha) $R_s$ (您也可以直接获取个股特异收益率与权重相乘直接进行计算)，但现实中受到仓位，调仓时间，费用等其他因素的影响，此公式并非完全成立的，AttributionAnalysis 中是使用做差的方式来计算特异收益率，即:

$$
R_s = R_P - \sum_{j=1}^{k} R^p_j \cdot f_j
$$

### 以指数作为基准的归因分析
- jqdatasdk 已经根据指数权重计算好了指数的风格暴露 $X^B$，可通过<span style="color:red;">`jqdatasdk.get_index_style_exposure`</span> 获取

投资组合 P 相对于指数的第 j 个因子的暴露可表示为 :

$$
X^{P2B}_j = X^P_j -  X^B_j
$$

- $X^{P2B}_j$ 可通过<span style="color:red;"> `jqfactor_analyzer.AttributionAnalysis().get_exposure2bench(index_symbol)` </span>获取

投资组合在第 j 个因子上相对于指数获取到的收益率可以表示为 :

$$
R^{P2B}_j =  R^P_j  -  R^B_j = X^P_j \cdot f_j  - X^B_j \cdot f_j  = f_j \cdot X^{P2B}_j
$$

在 AttributionAnalysis 中，风格及行业因子部分，将指数的仓位和持仓的仓位进行了对齐；同时考虑了现金产生的收益 (国家因子在仓位对齐后不会产生暴露收益，现金收益为 0，现金相对于指数的收益即为：(-1) × 剩余仓位 × 指数收益)

所以投资组合相对于指数的收益可以被表示为:

$$
R_{P2B} = \sum_{j=1}^{k}  R^{P2B}_j  + R^{P2B}_s + 现金相对于指数的收益
$$

- $R_{P2B}$ 等可通过 <span style="color:red;">`jqfactor_analyzer.AttributionAnalysis().get_attr_daily_returns2bench(index_symbol)` </span>获取

### 累积收益的处理
上述 `attr_daily_return` 和  `get_attr_daily_returns2bench(index_symbol)` 获取到的均为单日收益率，在计算累积收益时需要考虑复利影响。

$$
N_t =  \prod_{t=1}^{n} (R^p_t+1)
$$
$$
Rcum^p_{jt}  = N_{t-1} \cdot R^P_{jt}
$$

其中 :

- $N_t$ 为投资组合在第 t 天盘后的净值
- $R^p_t$ 为投资组合在第 t 天的日度收益率
- $Rcum^p_{jt}$ 为投资组合 p 的第 j 个因子在 t 日的累积收益
- $R^P_{jt}$ 为投资组合 p 的第 j 个因子在 t 日的日收益率
- $N_t,  Rcum^p_{jt}$ 均可通过<span style="color:red;"> `jqfactor_analyzer.AttributionAnalysis().attr_returns`</span> 获取
- 相对于基准的累积收益算法类似, 可通过 <span style="color:red;">`jqfactor_analyzer.AttributionAnalysis().get_attr_returns2bench` </span>获取

### 导入模块并登陆 jqdatasdk


```python
import jqdatasdk
import jqfactor_analyzer as ja
# 获取 jqdatasdk 授权，输入用户名、密码，申请地址：https://www.joinquant.com/default/index/sdk
# 聚宽官网，使用方法参见：https://www.joinquant.com/help/api/doc?name=JQDatadoc
jqdatasdk.auth("账号", "密码")
```

### 处理权重信息
此处使用的是 jqfactor_analyzer 提供的示例文件
数据格式要求 :
- 权重数据, 一个 dataframe, index 为日期, columns 为标的代码 (可使用 jqdatasdk.normalize_code 转为支持的格式), values 为权重, 每日的权重和应该小于 1
- 组合的日度收益数据, 一个 series, index 为日期, values 为日收益率


```python
import os
import pandas as pd
weight_path = os.path.join(os.path.dirname(ja.__file__), 'sample_data', 'weight_info.csv')
weight_infos = pd.read_csv(weight_path, index_col=0)
daily_return = weight_infos.pop("return")
```


```python
weight_infos.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    dataframe tbody tr th {
        vertical-align: top;
    }
    .dataframe thead th {
        text-align: right;
    }
</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>000006.XSHE</th>
      <th>000008.XSHE</th>
      <th>000009.XSHE</th>
      <th>000012.XSHE</th>
      <th>000021.XSHE</th>
      <th>000025.XSHE</th>
      <th>000027.XSHE</th>
      <th>000028.XSHE</th>
      <th>000031.XSHE</th>
      <th>000032.XSHE</th>
      <th>...</th>
      <th>603883.XSHG</th>
      <th>603885.XSHG</th>
      <th>603888.XSHG</th>
      <th>603893.XSHG</th>
      <th>603927.XSHG</th>
      <th>603939.XSHG</th>
      <th>603979.XSHG</th>
      <th>603983.XSHG</th>
      <th>605117.XSHG</th>
      <th>605358.XSHG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>0.000873</td>
      <td>0.001244</td>
      <td>0.002934</td>
      <td>0.001219</td>
      <td>0.001614</td>
      <td>0.000433</td>
      <td>0.001274</td>
      <td>0.001181</td>
      <td>0.001471</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.001294</td>
      <td>0.001536</td>
      <td>0.000781</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.001896</td>
      <td>NaN</td>
      <td>0.000482</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>0.000897</td>
      <td>0.001247</td>
      <td>0.002679</td>
      <td>0.001203</td>
      <td>0.001708</td>
      <td>0.000432</td>
      <td>0.001293</td>
      <td>0.001195</td>
      <td>0.001463</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.001298</td>
      <td>0.001505</td>
      <td>0.000824</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.001912</td>
      <td>NaN</td>
      <td>0.000466</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>0.000879</td>
      <td>0.001216</td>
      <td>0.002926</td>
      <td>0.001225</td>
      <td>0.001613</td>
      <td>0.000434</td>
      <td>0.001278</td>
      <td>0.001228</td>
      <td>0.001429</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.001238</td>
      <td>0.001534</td>
      <td>0.000767</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.001962</td>
      <td>NaN</td>
      <td>0.000488</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>0.000883</td>
      <td>0.001241</td>
      <td>0.002591</td>
      <td>0.001220</td>
      <td>0.001536</td>
      <td>0.000439</td>
      <td>0.001294</td>
      <td>0.001195</td>
      <td>0.001488</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.001267</td>
      <td>0.001575</td>
      <td>0.000764</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.001959</td>
      <td>NaN</td>
      <td>0.000468</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>0.000877</td>
      <td>0.001231</td>
      <td>0.002758</td>
      <td>0.001205</td>
      <td>0.001528</td>
      <td>0.000429</td>
      <td>0.001270</td>
      <td>0.001208</td>
      <td>0.001448</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.001277</td>
      <td>0.001554</td>
      <td>0.000749</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.001987</td>
      <td>NaN</td>
      <td>0.000474</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 818 columns</p>
</div>




```python
weight_infos.sum(axis=1).head(5)
```




    2020-01-02    0.752196
    2020-01-03    0.750206
    2020-01-06    0.752375
    2020-01-07    0.752054
    2020-01-08    0.748039
    dtype: float64



### 进行归因分析
**具体用法请查看[API文档](https://github.com/JoinQuant/jqfactor_analyzer/blob/master/docs/API%E6%96%87%E6%A1%A3.md), 此处仅作示例**


```python
An = ja.AttributionAnalysis(weight_infos, daily_return, style_type='style', industry='sw_l1', use_cn=True, show_data_progress=True)
```

    check/save factor cache : 100%|██████████| 54/54 [00:02<00:00, 25.75it/s]
    calc_style_exposure : 100%|██████████| 1087/1087 [00:27<00:00, 39.52it/s]
    calc_industry_exposure : 100%|██████████| 1087/1087 [00:19<00:00, 56.53it/s]



```python
An.exposure_portfolio.head(5) #查看暴露
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>beta</th>
      <th>momentum</th>
      <th>residual_volatility</th>
      <th>non_linear_size</th>
      <th>book_to_price_ratio</th>
      <th>liquidity</th>
      <th>earnings_yield</th>
      <th>growth</th>
      <th>leverage</th>
      <th>...</th>
      <th>801050</th>
      <th>801040</th>
      <th>801780</th>
      <th>801970</th>
      <th>801120</th>
      <th>801790</th>
      <th>801760</th>
      <th>801890</th>
      <th>801960</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>-0.487816</td>
      <td>0.468947</td>
      <td>-0.048262</td>
      <td>0.104597</td>
      <td>0.976877</td>
      <td>-0.112042</td>
      <td>0.278131</td>
      <td>-0.311944</td>
      <td>-0.000541</td>
      <td>-0.356787</td>
      <td>...</td>
      <td>0.030234</td>
      <td>0.023728</td>
      <td>0.010499</td>
      <td>NaN</td>
      <td>0.017049</td>
      <td>0.032292</td>
      <td>0.042405</td>
      <td>0.027871</td>
      <td>NaN</td>
      <td>0.752196</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>-0.485128</td>
      <td>0.461138</td>
      <td>-0.044422</td>
      <td>0.104270</td>
      <td>0.970710</td>
      <td>-0.110196</td>
      <td>0.271739</td>
      <td>-0.314469</td>
      <td>-0.002360</td>
      <td>-0.354623</td>
      <td>...</td>
      <td>0.030574</td>
      <td>0.023712</td>
      <td>0.010610</td>
      <td>NaN</td>
      <td>0.017071</td>
      <td>0.033261</td>
      <td>0.041491</td>
      <td>0.027631</td>
      <td>NaN</td>
      <td>0.750206</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>-0.477658</td>
      <td>0.464642</td>
      <td>-0.034905</td>
      <td>0.116226</td>
      <td>0.958563</td>
      <td>-0.118501</td>
      <td>0.277993</td>
      <td>-0.320429</td>
      <td>-0.001766</td>
      <td>-0.352186</td>
      <td>...</td>
      <td>0.030807</td>
      <td>0.023681</td>
      <td>0.010619</td>
      <td>NaN</td>
      <td>0.016953</td>
      <td>0.033203</td>
      <td>0.042406</td>
      <td>0.027906</td>
      <td>NaN</td>
      <td>0.752375</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>-0.474913</td>
      <td>0.456438</td>
      <td>-0.030596</td>
      <td>0.118867</td>
      <td>0.953152</td>
      <td>-0.117436</td>
      <td>0.274219</td>
      <td>-0.315071</td>
      <td>-0.000874</td>
      <td>-0.350100</td>
      <td>...</td>
      <td>0.030140</td>
      <td>0.024215</td>
      <td>0.010716</td>
      <td>NaN</td>
      <td>0.017240</td>
      <td>0.033022</td>
      <td>0.042867</td>
      <td>0.027853</td>
      <td>NaN</td>
      <td>0.752054</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>-0.474413</td>
      <td>0.452745</td>
      <td>-0.026417</td>
      <td>0.123923</td>
      <td>0.951369</td>
      <td>-0.115294</td>
      <td>0.271193</td>
      <td>-0.305295</td>
      <td>-0.000920</td>
      <td>-0.345431</td>
      <td>...</td>
      <td>0.030176</td>
      <td>0.023694</td>
      <td>0.010671</td>
      <td>NaN</td>
      <td>0.017303</td>
      <td>0.032777</td>
      <td>0.040977</td>
      <td>0.027820</td>
      <td>NaN</td>
      <td>0.748039</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 43 columns</p>
</div>




```python
An.attr_daily_returns.head(5) #查看日度收益拆解
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>beta</th>
      <th>momentum</th>
      <th>residual_volatility</th>
      <th>non_linear_size</th>
      <th>book_to_price_ratio</th>
      <th>liquidity</th>
      <th>earnings_yield</th>
      <th>growth</th>
      <th>leverage</th>
      <th>...</th>
      <th>801970</th>
      <th>801120</th>
      <th>801790</th>
      <th>801760</th>
      <th>801890</th>
      <th>801960</th>
      <th>country</th>
      <th>common_return</th>
      <th>specific_return</th>
      <th>total_return</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>0.000241</td>
      <td>-0.000144</td>
      <td>0.000130</td>
      <td>0.000090</td>
      <td>0.000955</td>
      <td>-0.000039</td>
      <td>-0.000045</td>
      <td>0.000174</td>
      <td>7.907650e-08</td>
      <td>0.000148</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.000168</td>
      <td>-0.000019</td>
      <td>0.000500</td>
      <td>-0.000050</td>
      <td>NaN</td>
      <td>0.000860</td>
      <td>0.003030</td>
      <td>-0.001083</td>
      <td>0.001948</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>-0.000014</td>
      <td>0.000151</td>
      <td>0.000119</td>
      <td>0.000199</td>
      <td>0.002035</td>
      <td>-0.000017</td>
      <td>0.000025</td>
      <td>0.000573</td>
      <td>-1.457480e-07</td>
      <td>0.000160</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.000178</td>
      <td>-0.000145</td>
      <td>0.000286</td>
      <td>0.000015</td>
      <td>NaN</td>
      <td>0.000949</td>
      <td>0.004990</td>
      <td>0.002358</td>
      <td>0.007348</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>0.000176</td>
      <td>0.001208</td>
      <td>0.000002</td>
      <td>0.000236</td>
      <td>0.001533</td>
      <td>0.000012</td>
      <td>-0.000213</td>
      <td>-0.000627</td>
      <td>8.726552e-07</td>
      <td>0.000250</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.000077</td>
      <td>-0.000003</td>
      <td>0.000834</td>
      <td>-0.000008</td>
      <td>NaN</td>
      <td>0.006875</td>
      <td>0.009541</td>
      <td>-0.000621</td>
      <td>0.008920</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>-0.000190</td>
      <td>-0.001919</td>
      <td>-0.000007</td>
      <td>0.000019</td>
      <td>0.000199</td>
      <td>0.000027</td>
      <td>-0.000134</td>
      <td>0.000400</td>
      <td>-8.393073e-09</td>
      <td>-0.000140</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.000038</td>
      <td>-0.000384</td>
      <td>-0.000414</td>
      <td>0.000104</td>
      <td>NaN</td>
      <td>-0.009655</td>
      <td>-0.010019</td>
      <td>-0.000516</td>
      <td>-0.010535</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 46 columns</p>
</div>




```python
An.attr_returns.head(5)  #查看累积收益
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>beta</th>
      <th>momentum</th>
      <th>residual_volatility</th>
      <th>non_linear_size</th>
      <th>book_to_price_ratio</th>
      <th>liquidity</th>
      <th>earnings_yield</th>
      <th>growth</th>
      <th>leverage</th>
      <th>...</th>
      <th>801970</th>
      <th>801120</th>
      <th>801790</th>
      <th>801760</th>
      <th>801890</th>
      <th>801960</th>
      <th>country</th>
      <th>common_return</th>
      <th>specific_return</th>
      <th>total_return</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>0.000241</td>
      <td>-0.000144</td>
      <td>0.000130</td>
      <td>0.000090</td>
      <td>0.000955</td>
      <td>-0.000039</td>
      <td>-0.000045</td>
      <td>0.000174</td>
      <td>7.907650e-08</td>
      <td>0.000148</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.000168</td>
      <td>-0.000019</td>
      <td>0.000500</td>
      <td>-0.000050</td>
      <td>NaN</td>
      <td>0.000860</td>
      <td>0.003030</td>
      <td>-0.001083</td>
      <td>0.001948</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>0.000227</td>
      <td>0.000007</td>
      <td>0.000249</td>
      <td>0.000290</td>
      <td>0.002994</td>
      <td>-0.000056</td>
      <td>-0.000020</td>
      <td>0.000748</td>
      <td>-6.695534e-08</td>
      <td>0.000308</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.000346</td>
      <td>-0.000164</td>
      <td>0.000787</td>
      <td>-0.000035</td>
      <td>NaN</td>
      <td>0.001812</td>
      <td>0.008030</td>
      <td>0.001280</td>
      <td>0.009310</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>0.000405</td>
      <td>0.001226</td>
      <td>0.000252</td>
      <td>0.000528</td>
      <td>0.004541</td>
      <td>-0.000044</td>
      <td>-0.000234</td>
      <td>0.000115</td>
      <td>8.138242e-07</td>
      <td>0.000560</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.000268</td>
      <td>-0.000168</td>
      <td>0.001629</td>
      <td>-0.000043</td>
      <td>NaN</td>
      <td>0.008750</td>
      <td>0.017660</td>
      <td>0.000653</td>
      <td>0.018313</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>0.000212</td>
      <td>-0.000728</td>
      <td>0.000245</td>
      <td>0.000547</td>
      <td>0.004744</td>
      <td>-0.000016</td>
      <td>-0.000371</td>
      <td>0.000522</td>
      <td>8.052775e-07</td>
      <td>0.000418</td>
      <td>...</td>
      <td>NaN</td>
      <td>-0.000229</td>
      <td>-0.000559</td>
      <td>0.001207</td>
      <td>0.000064</td>
      <td>NaN</td>
      <td>-0.001081</td>
      <td>0.007457</td>
      <td>0.000128</td>
      <td>0.007585</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 46 columns</p>
</div>




```python
An.get_attr_returns2bench('000905.XSHG').head(5)  #查看相对指数的累积收益
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>size</th>
      <th>beta</th>
      <th>momentum</th>
      <th>residual_volatility</th>
      <th>non_linear_size</th>
      <th>book_to_price_ratio</th>
      <th>liquidity</th>
      <th>earnings_yield</th>
      <th>growth</th>
      <th>leverage</th>
      <th>...</th>
      <th>801970</th>
      <th>801120</th>
      <th>801790</th>
      <th>801760</th>
      <th>801890</th>
      <th>801960</th>
      <th>common_return</th>
      <th>cash</th>
      <th>specific_return</th>
      <th>total_return</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>2.247752e-08</td>
      <td>5.274612e-07</td>
      <td>-1.010579e-06</td>
      <td>-1.780933e-07</td>
      <td>-5.018849e-09</td>
      <td>-1.576053e-07</td>
      <td>1.815168e-07</td>
      <td>3.067299e-07</td>
      <td>-8.676436e-08</td>
      <td>3.843589e-07</td>
      <td>...</td>
      <td>NaN</td>
      <td>-8.367921e-07</td>
      <td>2.211999e-07</td>
      <td>2.512287e-07</td>
      <td>-2.031997e-07</td>
      <td>NaN</td>
      <td>-0.000006</td>
      <td>-0.000670</td>
      <td>-0.000079</td>
      <td>-0.000755</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>3.139000e-09</td>
      <td>-2.167887e-06</td>
      <td>1.005890e-06</td>
      <td>-9.837778e-06</td>
      <td>1.803920e-06</td>
      <td>3.592758e-07</td>
      <td>-3.082887e-07</td>
      <td>-4.489268e-06</td>
      <td>-1.570012e-07</td>
      <td>-7.565016e-07</td>
      <td>...</td>
      <td>NaN</td>
      <td>-4.620739e-06</td>
      <td>-2.607788e-06</td>
      <td>-8.734669e-06</td>
      <td>-1.166518e-07</td>
      <td>NaN</td>
      <td>-0.000063</td>
      <td>-0.003198</td>
      <td>-0.000234</td>
      <td>-0.003494</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>-5.129552e-08</td>
      <td>-2.485408e-05</td>
      <td>9.140735e-07</td>
      <td>-2.227106e-05</td>
      <td>1.453669e-06</td>
      <td>-5.066033e-08</td>
      <td>4.500972e-06</td>
      <td>4.348111e-06</td>
      <td>1.794315e-07</td>
      <td>-3.707358e-06</td>
      <td>...</td>
      <td>NaN</td>
      <td>-1.876927e-06</td>
      <td>-2.703177e-06</td>
      <td>-3.476170e-05</td>
      <td>-2.429496e-07</td>
      <td>NaN</td>
      <td>-0.000095</td>
      <td>-0.006224</td>
      <td>-0.000283</td>
      <td>-0.006603</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>-1.236180e-07</td>
      <td>4.020758e-05</td>
      <td>1.082783e-06</td>
      <td>-2.386474e-05</td>
      <td>1.502709e-06</td>
      <td>-1.806807e-06</td>
      <td>1.001751e-05</td>
      <td>-7.241071e-06</td>
      <td>1.893800e-07</td>
      <td>-1.425501e-06</td>
      <td>...</td>
      <td>NaN</td>
      <td>2.019730e-07</td>
      <td>-1.379156e-05</td>
      <td>-1.232299e-05</td>
      <td>1.799073e-06</td>
      <td>NaN</td>
      <td>-0.000087</td>
      <td>-0.002647</td>
      <td>-0.000427</td>
      <td>-0.003160</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 46 columns</p>
</div>




```python
An.plot_exposure(factors='style',index_symbol=None,figsize=(15,7))
```


![Img](https://image.joinquant.com/4e362f3cbce335f035e568d4a264ce7d)



```python
An.plot_returns(factors='style',index_symbol=None,figsize=(15,7))
```


![Img](https://image.joinquant.com/33494d30d39798500f5cdd7d560256d1)



```python
An.plot_exposure_and_returns(factors='style',index_symbol=None,show_factor_perf=False,figsize=(12,6))
```


![Img](https://image.joinquant.com/92359b7544f550d912bcaf602da10238)


## 因子数据本地缓存使用示例
**具体用法请查看[API文档](https://github.com/JoinQuant/jqfactor_analyzer/blob/master/docs/API%E6%96%87%E6%A1%A3.md), 此处仅作示例**

### 设置缓存目录


```python
from jqfactor_analyzer.factor_cache import set_cache_dir,get_cache_dir
# my_path = 'E:\\jqfactor_cache'
# set_cache_dir(my_path) #设置缓存目录为my_path
print(get_cache_dir()) #输出缓存目录
```

    C:\Users\wq\jqfactor_datacache\bundle


### 缓存/检查缓存和读取已缓存数据


```python
from jqfactor_analyzer.factor_cache import save_factor_values_by_group,get_factor_values_by_cache,get_factor_folder,get_cache_dir
# import jqdatasdk as jq
# jq.auth("账号",'密码') #登陆jqdatasdk来从服务端缓存数据

all_factors = jqdatasdk.get_all_factors()
factor_names = all_factors[all_factors.category=='growth'].factor.tolist()  #将聚宽因子库中的成长类因子作为一组因子
group_name = 'growth_factors' #因子组名定义为'growth_factors'
start_date = '2021-01-01'
end_date = '2021-06-01'
# 检查/缓存因子数据
factor_path = save_factor_values_by_group(start_date,end_date,factor_names=factor_names,group_name=group_name,overwrite=False,show_progress=True)
# factor_path = os.path.join(get_cache_dir(), get_factor_folder(factor_names,group_name=group_name)  #等同于save_factor_values_by_group返回的路径

```

    check/save factor cache : 100%|██████████| 6/6 [00:01<00:00,  5.87it/s]



```python
# 循环获取缓存的因子数据,并拼接
trade_days = jqdatasdk.get_trade_days(start_date,end_date)
factor_values = {}
for date in trade_days:
    factor_values[date] = get_factor_values_by_cache(date,codes=None,factor_names=factor_names,group_name=group_name, factor_path=factor_path)#这里实际只需要指定group_name,factor_names参数的其中一个,缓存时指定了group_name时,factor_names不生效
factor_values = pd.concat(factor_values)
factor_values.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>financing_cash_growth_rate</th>
      <th>net_asset_growth_rate</th>
      <th>net_operate_cashflow_growth_rate</th>
      <th>net_profit_growth_rate</th>
      <th>np_parent_company_owners_growth_rate</th>
      <th>operating_revenue_growth_rate</th>
      <th>PEG</th>
      <th>total_asset_growth_rate</th>
      <th>total_profit_growth_rate</th>
    </tr>
    <tr>
      <th></th>
      <th>code</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">2021-01-04</th>
      <th>000001.XSHE</th>
      <td>4.218607</td>
      <td>0.245417</td>
      <td>-3.438636</td>
      <td>-0.036129</td>
      <td>-0.036129</td>
      <td>0.139493</td>
      <td>NaN</td>
      <td>0.172409</td>
      <td>-0.053686</td>
    </tr>
    <tr>
      <th>000002.XSHE</th>
      <td>-1.059306</td>
      <td>0.236022</td>
      <td>0.266020</td>
      <td>0.009771</td>
      <td>0.064828</td>
      <td>0.115457</td>
      <td>1.229423</td>
      <td>0.107217</td>
      <td>-0.013790</td>
    </tr>
    <tr>
      <th>000004.XSHE</th>
      <td>NaN</td>
      <td>11.430834</td>
      <td>-0.019530</td>
      <td>-3.350306</td>
      <td>-3.551808</td>
      <td>-0.328126</td>
      <td>NaN</td>
      <td>10.912087</td>
      <td>-3.888289</td>
    </tr>
    <tr>
      <th>000005.XSHE</th>
      <td>-1.014341</td>
      <td>0.052103</td>
      <td>-2.331018</td>
      <td>-0.480705</td>
      <td>-0.461062</td>
      <td>-0.700859</td>
      <td>NaN</td>
      <td>-0.040798</td>
      <td>-0.567470</td>
    </tr>
    <tr>
      <th>000006.XSHE</th>
      <td>-0.978757</td>
      <td>0.112236</td>
      <td>-1.509728</td>
      <td>0.083089</td>
      <td>0.044869</td>
      <td>0.170041</td>
      <td>1.931730</td>
      <td>-0.005611</td>
      <td>0.113066</td>
    </tr>
  </tbody>
</table>
</div>



## 单因子分析使用示例
**具体用法请查看[API文档](https://github.com/JoinQuant/jqfactor_analyzer/blob/master/docs/API%E6%96%87%E6%A1%A3.md), 此处仅作示例**
### 示例：5日平均换手率因子分析


```python
# 载入函数库
import pandas as pd
import jqfactor_analyzer as ja

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

    check/save price cache : 100%|██████████| 13/13 [00:00<00:00, 25.60it/s]
    load price info : 100%|██████████| 253/253 [00:06<00:00, 38.09it/s]
    load industry info : 100%|██████████| 243/243 [00:00<00:00, 331.46it/s]





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>period_1</th>
      <th>period_10</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-02</th>
      <td>0.141204</td>
      <td>-0.058936</td>
    </tr>
    <tr>
      <th>2018-01-03</th>
      <td>0.082738</td>
      <td>-0.176327</td>
    </tr>
    <tr>
      <th>2018-01-04</th>
      <td>-0.183788</td>
      <td>-0.196901</td>
    </tr>
    <tr>
      <th>2018-01-05</th>
      <td>0.057023</td>
      <td>-0.180102</td>
    </tr>
    <tr>
      <th>2018-01-08</th>
      <td>-0.025403</td>
      <td>-0.187145</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2018-12-24</th>
      <td>0.098161</td>
      <td>-0.198127</td>
    </tr>
    <tr>
      <th>2018-12-25</th>
      <td>-0.269072</td>
      <td>-0.166092</td>
    </tr>
    <tr>
      <th>2018-12-26</th>
      <td>-0.430034</td>
      <td>-0.117108</td>
    </tr>
    <tr>
      <th>2018-12-27</th>
      <td>-0.107514</td>
      <td>-0.040684</td>
    </tr>
    <tr>
      <th>2018-12-28</th>
      <td>-0.013224</td>
      <td>0.039446</td>
    </tr>
  </tbody>
</table>
<p>243 rows × 2 columns</p>
</div>




```python
# 生成统计图表
far.create_full_tear_sheet(
    demeaned=False, group_adjust=False, by_group=False,
    turnover_periods=None, avgretplot=(5, 15), std_bar=False
)
```

    分位数统计



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>std</th>
      <th>count</th>
      <th>count %</th>
    </tr>
    <tr>
      <th>factor_quantile</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.00000</td>
      <td>0.30046</td>
      <td>0.072019</td>
      <td>0.056611</td>
      <td>7293</td>
      <td>10.054595</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.08846</td>
      <td>0.49034</td>
      <td>0.198844</td>
      <td>0.066169</td>
      <td>7266</td>
      <td>10.017371</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.14954</td>
      <td>0.65984</td>
      <td>0.309961</td>
      <td>0.089310</td>
      <td>7219</td>
      <td>9.952574</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.22594</td>
      <td>0.80136</td>
      <td>0.423978</td>
      <td>0.111141</td>
      <td>7248</td>
      <td>9.992555</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.30904</td>
      <td>0.99400</td>
      <td>0.553684</td>
      <td>0.133578</td>
      <td>7280</td>
      <td>10.036672</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.38860</td>
      <td>1.23760</td>
      <td>0.696531</td>
      <td>0.166341</td>
      <td>7211</td>
      <td>9.941545</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.48394</td>
      <td>1.56502</td>
      <td>0.874488</td>
      <td>0.204828</td>
      <td>7240</td>
      <td>9.981526</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.61900</td>
      <td>2.09560</td>
      <td>1.132261</td>
      <td>0.265739</td>
      <td>7226</td>
      <td>9.962225</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.84984</td>
      <td>3.30790</td>
      <td>1.639863</td>
      <td>0.436992</td>
      <td>7261</td>
      <td>10.010478</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.23172</td>
      <td>40.47726</td>
      <td>4.276270</td>
      <td>3.640945</td>
      <td>7290</td>
      <td>10.050459</td>
    </tr>
  </tbody>
</table>
</div>


​
​    -------------------------
​
​    收益分析



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>period_1</th>
      <th>period_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ann. alpha</th>
      <td>-0.087</td>
      <td>-0.060</td>
    </tr>
    <tr>
      <th>beta</th>
      <td>1.218</td>
      <td>1.238</td>
    </tr>
    <tr>
      <th>Mean Period Wise Return Top Quantile (bps)</th>
      <td>-20.913</td>
      <td>-18.530</td>
    </tr>
    <tr>
      <th>Mean Period Wise Return Bottom Quantile (bps)</th>
      <td>-6.156</td>
      <td>-6.452</td>
    </tr>
    <tr>
      <th>Mean Period Wise Spread (bps)</th>
      <td>-14.757</td>
      <td>-13.177</td>
    </tr>
  </tbody>
</table>
</div>



    <Figure size 640x480 with 0 Axes>



![Img](https://image.joinquant.com/5669a6a708055c73c4bc443677f21344)



    <Figure size 640x480 with 0 Axes>



![Img](https://image.joinquant.com/d336030c71a3cbf9d2ef56fdd8757ba5)



......(图片过多,此处内容演示已省略,请参考api说明使用)



![Img](https://image.joinquant.com/ce00c434033ac68259374438cb10ec06)


​
​    -------------------------
​
​    IC 分析



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>period_1</th>
      <th>period_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>IC Mean</th>
      <td>-0.030</td>
      <td>-0.085</td>
    </tr>
    <tr>
      <th>IC Std.</th>
      <td>0.213</td>
      <td>0.176</td>
    </tr>
    <tr>
      <th>IR</th>
      <td>-0.140</td>
      <td>-0.487</td>
    </tr>
    <tr>
      <th>t-stat(IC)</th>
      <td>-2.180</td>
      <td>-7.587</td>
    </tr>
    <tr>
      <th>p-value(IC)</th>
      <td>0.030</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>IC Skew</th>
      <td>0.240</td>
      <td>0.091</td>
    </tr>
    <tr>
      <th>IC Kurtosis</th>
      <td>-0.420</td>
      <td>-0.485</td>
    </tr>
  </tbody>
</table>
</div>



    <Figure size 640x480 with 0 Axes>



![Img](https://image.joinquant.com/07aad6b961c5c38fa62b2d4601e49acb)



    <Figure size 640x480 with 0 Axes>



​
​    -------------------------
​
​    换手率分析



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>period_1</th>
      <th>period_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Quantile 1 Mean Turnover</th>
      <td>0.055</td>
      <td>0.222</td>
    </tr>
    <tr>
      <th>Quantile 2 Mean Turnover</th>
      <td>0.136</td>
      <td>0.447</td>
    </tr>
    <tr>
      <th>Quantile 3 Mean Turnover</th>
      <td>0.206</td>
      <td>0.599</td>
    </tr>
    <tr>
      <th>Quantile 4 Mean Turnover</th>
      <td>0.268</td>
      <td>0.680</td>
    </tr>
    <tr>
      <th>Quantile 5 Mean Turnover</th>
      <td>0.307</td>
      <td>0.730</td>
    </tr>
    <tr>
      <th>Quantile 6 Mean Turnover</th>
      <td>0.337</td>
      <td>0.742</td>
    </tr>
    <tr>
      <th>Quantile 7 Mean Turnover</th>
      <td>0.326</td>
      <td>0.735</td>
    </tr>
    <tr>
      <th>Quantile 8 Mean Turnover</th>
      <td>0.279</td>
      <td>0.708</td>
    </tr>
    <tr>
      <th>Quantile 9 Mean Turnover</th>
      <td>0.196</td>
      <td>0.593</td>
    </tr>
    <tr>
      <th>Quantile 10 Mean Turnover</th>
      <td>0.073</td>
      <td>0.283</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>period_1</th>
      <th>period_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Mean Factor Rank Autocorrelation</th>
      <td>0.991</td>
      <td>0.884</td>
    </tr>
  </tbody>
</table>
</div>


......(图片过多,此处内容演示已省略,请参考api说明使用)


###  获取聚宽因子库数据的方法
[聚宽因子库](https://www.joinquant.com/help/api/help#name:factor_values)包含数百个质量、情绪、风险等其他类目的因子

连接jqdatasdk获取数据包，数据接口需调用聚宽 [jqdatasdk](https://www.joinquant.com/help/api/doc?name=JQDatadoc) 接口获取金融数据 ([试用注册地址](https://www.joinquant.com/default/index/sdk))


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

### 将自有因子值转换成 DataFrame 格式的数据
- index 为日期，格式为 pandas 日期通用的 DatetimeIndex

- columns 为股票代码，格式要求符合聚宽的代码定义规则（如：平安银行的股票代码为 000001.XSHE）

  - 如果是深交所上市的股票，在股票代码后面需要加入 .XSHE
  - 如果是上交所上市的股票，在股票代码后面需要加入 .XSHG
- 将 pandas.DataFrame 转换成满足格式要求数据格式

  首先要保证 index 为 DatetimeIndex 格式，一般是通过 pandas 提供的 pandas.to_datetime 函数进行转换，在转换前应确保 index 中的值都为合理的日期格式， 如 '2018-01-01' / '20180101'，之后再调用 pandas.to_datetime 进行转换；另外应确保 index 的日期是按照从小到大的顺序排列的，可以通过 sort_index 进行排序；最后请检查 columns 中的股票代码是否都满足聚宽的代码定义。

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
- 将键为日期，值为各股票因子值的 Series 的 dict 转换成 pandas.DataFrame，可以直接利用 pandas.DataFrame 生成
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
# 之后请按照 DataFrae 的方法转换成满足格式要求数据格式
```

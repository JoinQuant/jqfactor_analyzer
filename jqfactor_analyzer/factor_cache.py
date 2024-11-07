import hashlib
from itertools import groupby
import pandas as pd
import os
import json
import functools
import logging
from .when import today, now, TimeDelta
from tqdm import tqdm


try:
    import jqdata
    api = jqdata.apis
    api_name = 'jqdata'
except ImportError:
    import jqdatasdk
    api = jqdatasdk
    api_name = 'jqdatasdk'


def get_cache_config():
    """获取缓存目录"""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'config.json'
    )
    if not os.path.exists(config_path):
        return set_cache_dir("")
    else:
        with open(config_path, 'r') as conf:
            return json.load(conf)


def set_cache_dir(path):
    """设置缓存目录"""
    cfg = {'default_dir': '~/jqfactor_datacache/bundle',
           'user_dir': os.path.expanduser(path)}
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'config.json'
    )
    with open(config_path, 'w') as conf:
        json.dump(cfg, conf)
    get_cache_dir.cache_clear()
    return cfg


@functools.lru_cache()
def get_cache_dir():
    # 优先获取用户配置的缓存目录, 若无, 则使用默认目录
    cfg = get_cache_config()
    user_path = cfg.get('user_dir', "")
    if user_path != "":
        return os.path.expanduser(user_path)
    return os.path.expanduser(cfg['default_dir'])


def list_to_tuple_converter(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 将所有位置参数中的 list 转换为 tuple
        args = tuple(tuple(arg) if isinstance(
            arg, list) else arg for arg in args)

        # 将关键字参数中的 list 转换为 tuple
        kwargs = {k: tuple(v) if isinstance(v, list)
                  else v for k, v in kwargs.items()}

        return func(*args, **kwargs)
    return wrapper


@list_to_tuple_converter
@functools.lru_cache()
def get_factor_folder(factor_names, group_name=None):
    """获取因子组的文件夹
    factor_names : 因子名列表
    group_name : 因子组的名称, 如果指定则使用指定的名称作为文件夹名
                 否则用 jqfactor_cache_ + 因子名的 md5 值 (顺序无关) 作为文件夹名
    """
    if group_name:
        return group_name
    else:
        if factor_names == 'prices':
            return 'jqprice_cache'
        if isinstance(factor_names, str):
            factor_names = [factor_names]
        factor_names = sorted(factor_names)
        factor_names = ''.join(factor_names)
        hash_object = hashlib.md5(factor_names.encode())
        hash_hex = hash_object.hexdigest()
    return f"jqfactor_cache_{hash_hex}"


def get_date_miss_group(A, B):
    '''将A相比B缺失的部分按连续性进行分组'''
    group_values = []
    masks = [(x not in A) for x in B]
    for key, group in groupby(zip(B, masks), lambda x: x[1]):
        if key:
            group_values.append([item[0] for item in group])
    return group_values


def save_data_by_month(factor_names, start, end, month_path):
    """按时间段获取储存数据(不要跨月)
    """
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    security_info = api.get_all_securities()

    month_value = {}
    stocks = security_info[(security_info.start_date <= end) & (
        security_info.end_date >= start)].index.tolist()
    if factor_names == 'prices':
        month_value = api.get_price(stocks, start_date=start, end_date=end,
                                    skip_paused=False, round=False,
                                    fields=['open', 'close', 'factor'],
                                    fq='post', panel=False)
        if month_value.empty:
            return 0
        month_value.set_index(['code', 'time'], inplace=True)
        month_value[['open', 'close']] = month_value[[
            'open', 'close']].div(month_value['factor'], axis=0)
    else:
        for factor in factor_names:
            month_value.update(api.get_factor_values(stocks,
                                                     start_date=start,
                                                     end_date=end,
                                                     factors=factor))
        if not month_value:
            return 0
        month_value = pd.concat(month_value).unstack(level=1).T
    month_value.index.names = ('code', 'date')

    for date, data in month_value.groupby(month_value.index.get_level_values(1)):
        data = data.reset_index(level=1, drop=True)
        data = data.reindex(security_info[(security_info.start_date <= date) & (
            security_info.end_date >= date)].index.tolist())
        # 数据未产生, 或者已经生产了但是全为 nan
        if data.isna().values.all():
            continue
        path = os.path.join(month_path, date.strftime("%Y%m%d") + ".feather")
        data.reset_index().to_feather(path)
    return month_value


def save_factor_valeus_by_group(start_date, end_date,
                                factor_names='prices', group_name=None,
                                overwrite=False, cache_dir=None, show_progress=True):
    """将因子库数据按因子组储存到本地
    start_date : 开始时间
    end_date : 结束时间
    factor_names : 因子组所含因子的名称,除过因子库中支持的因子外，还支持指定为'prices'缓存价格数据
    overwrite  : 文件已存在时是否覆盖更新
    返回 : 因子组储存的路径 , 文件以天为单位储存,每天一个feather文件,每月一个文件夹,columns第一列是因子名称, 而后是当天在市的所有标的代码
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()

    start_date = pd.to_datetime(start_date).date()
    last_day = today() - TimeDelta(days=1) if now().hour > 8 else today() - TimeDelta(days=2)
    end_date = min(pd.to_datetime(end_date).date(), last_day)
    date_range = pd.date_range(start_date, end_date, freq='1m')
    _date = pd.to_datetime(end_date)
    if len(date_range) == 0 or date_range[-1] < _date:
        date_range = date_range.append(pd.Index([_date]))

    if show_progress:
        if isinstance(show_progress, str):
            desc = show_progress
        elif factor_names == 'prices':
            desc = 'check/save price cache '
        else:
            desc = 'check/save factor cache '
        date_range = tqdm(date_range, total=len(date_range), desc=desc)
    root_path = os.path.join(
        cache_dir, get_factor_folder(factor_names, group_name))

    for end in date_range:
        start = max(end.replace(day=1).date(), start_date)
        month_path = os.path.join(root_path, end.strftime("%Y%m"))
        if not os.path.exists(month_path):
            os.makedirs(month_path)
        elif not overwrite:
            dates = [x.split(".")[0] for x in os.listdir(month_path)]
            dates = pd.to_datetime(dates).date
            trade_days = api.get_trade_days(start, end)
            miss_group = get_date_miss_group(dates, trade_days)
            if miss_group:
                for group in miss_group:
                    save_data_by_month(
                        factor_names, group[0], group[-1], month_path)
            continue
        save_data_by_month(factor_names, start, end, month_path)

    return root_path


def get_factor_values_by_cache(date, codes=None, factor_names=None, factor_path=None):
    """从缓存的文件读取因子数据, 文件不存在时返回空的 DataFrame"""
    date = pd.to_datetime(date)
    if factor_path:
        path = os.path.join(factor_path,
                            date.strftime("%Y%m"),
                            date.strftime("%Y%m%d") + ".feather")
    elif factor_names:
        path = os.path.join(get_cache_dir(),
                            get_factor_folder(factor_names),
                            date.strftime("%Y%m"),
                            date.strftime("%Y%m%d") + ".feather")
    else:
        raise ValueError("factor_names 和 factor_path 至少指定其中一个")
    # 数据未产生, 或者已经生产了但是全为 nan
    if not os.path.exists(path):
        factor_names = factor_names if factor_names != 'prices' else [
            'open', 'close', 'factor']
        data = pd.DataFrame(index=codes, columns=factor_names)
        data.index.name = 'code'
        return data

    try:
        data = pd.read_feather(path, use_threads=False).set_index('code')
    except Exception as e:
        if factor_names:
            logging.error("\n{} 缓存文件可能已损坏, 请重新下载".format(date))
            save_data_by_month(factor_names,
                               date, date,
                               os.path.join(factor_path, date.strftime("%Y%m")))
            data = get_factor_values_by_cache(
                date, codes, factor_names, factor_path)
        else:
            raise ValueError(
                "\n{} 缓存文件可能已损坏, 请重新下载 (指定 factor_names 时会自动下载) {} ".format(date, e))

    if codes is not None:
        data = data.reindex(codes)

    return data

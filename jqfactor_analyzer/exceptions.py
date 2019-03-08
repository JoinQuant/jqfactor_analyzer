# -*- coding: utf-8 -*-


from functools import wraps


def rethrow(exception, additional_message):
    """
    重新抛出当前作用域中的最后一个异常, 保留堆栈信息, 并且在报错信息中添加其他内容
    """
    e = exception
    m = additional_message
    if not e.args:
        e.args = (m,)
    else:
        e.args = (e.args[0] + m,) + e.args[1:]
    raise e


def non_unique_bin_edges_error(func):
    """
    捕获 pd.qcut 的异常, 添加提示信息并报错
    """
    message = u"""
    根据输入的 quantiles 计算时发生错误.
    这通常发生在输入包含太多相同值, 使得它们跨越多个分位.
    每天的因子值是按照分位数平均分组的, 相同的值不能跨越多个分位数.
    可能的解决方法:
    1. 减少分位数
    2. 调整因子减少重复值
    3. 尝试不同的股票池
    """

    @wraps(func)
    def dec(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            if 'Bin edges must be unique' in str(e):
                rethrow(e, message)
            raise

    return dec


class MaxLossExceededError(Exception):
    pass

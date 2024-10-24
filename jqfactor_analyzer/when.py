# -*- coding: utf-8 -*-

import six
import datetime

import pandas as pd


DateTime = datetime.datetime
Date = datetime.date
Time = datetime.time
TimeDelta = datetime.timedelta

today = datetime.date.today
now = datetime.datetime.now


def date2str(date, format='%Y-%m-%d'):
    return pd.to_datetime(date).strftime(format)


def convert_date(date):
    if isinstance(date, six.string_types):
        if ':' in date:
            date = date[:10]
        return datetime.datetime.strptime(date, '%Y-%m-%d').date()
    elif isinstance(date, datetime.datetime):
        return date.date()
    elif isinstance(date, datetime.date):
        return date
    raise Exception("date 必须是datetime.date, datetime.datetime或者如下格式的字符串:'2015-01-05'")

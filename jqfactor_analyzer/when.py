# -*- coding: utf-8 -*-


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

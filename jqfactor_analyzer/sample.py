# -*- coding: utf-8 -*-

import os
import pandas as pd


VOL5 = pd.read_csv(
    os.path.abspath(os.path.join(os.path.dirname(__file__),
                                 'sample_data',
                                 'VOL5.csv')),
    header=0, index_col=0, encoding='utf-8'
)

VOL5.index = pd.to_datetime(VOL5.index)
VOL5.index.set_names(['date'], inplace=True)
VOL5.columns.set_names(['asset'], inplace=True)

#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np

# Data preprocessing
def preprocess_data(df):
    df.replace('unknown', np.nan, inplace=True)
    
    df2 = df[(df['y'] == 'no') & (df['default'] == 'no')]
    df2.dropna(inplace=True)
    
    df3 = df[(df['y'] == 'yes')& (df['default'] == 'no')]
    df3.fillna(df3.mode().iloc[0], inplace=True)
    
    df4 = df[df['default'] == 'yes']
    df4.fillna(df4.mode().iloc[0], inplace=True)

    df5 = pd.concat([df2, df3, df4], axis=0)
    df5['default'].value_counts()

    df6 = pd.get_dummies(df5, drop_first=True)
    
    return df6


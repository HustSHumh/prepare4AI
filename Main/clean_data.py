import pandas as pd 
import numpy as np
import datetime
import os

import cfg

df = pd.DataFrame()

# 不变值监测
def get_unchange_data():
    no_change_windows_size = 10
    df['same'] = df['num'] == df['num'].shift(1)
    for idx, value in enumerate(df['same'].values):
        if value == False:
            df['same'][idx] = 0
            continue
        if idx == 0:
            df['same'][idx] = 1
        else:
            df['same'][idx] = df['same'][idx-1] + 1
    df['same_'] = df['same'][df['same'] > no_change_windows_size]
    return df['same_']

# 平滑处理
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')



# 处理数据
def clean_data():
    # 数据转换，日期转换
    df['datetime'] = pd.to_datetime(df['datetime'], format=f'%Y-%m-%d %H:%M%S').tz_localize('Asia/shanghai')
    # 数据转换，浮点数转换
    df['num'] = pd.to_numeric(df['num'], errors='coerce')

    df['num'] = df['num'].apply(lambda x: round(x, 4))
    # 最大值获取，最大值剔除
    df['num'] = df['num'].nlargest(3)
    df['num'] = df['num'].apply(lambda x: x if x > 1 else np.nan)

    # 最小值获取，最小值剔除
    df['num'] = df['num'].nsmallest(3)
    df['num'] = df['num'].apply(lambda x: x if x < 1 else np.nan)
    
    # 跳变值监测

    # 数据列的drop
    df.drop(columns=['col'], inplace=True)

    # 数据插值
    # 1、线性插值
    df['num'] = df['num'].interpolate(limit=3)
    # 2、填充
    df['num'] = df['num'].ffill()


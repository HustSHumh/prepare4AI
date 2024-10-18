import pandas as pd 
import numpy as np
import datetime
import os

import cfg

df = pd.DataFrame()

# 处理数据
def clean_data():
    # 数据转换，日期转换
    df['datetime'] = pd.to_datetime(df['datetime'], format=f'%Y-%m-%d %H:%M%S').tz_localize('Asia/shanghai')
    # 数据转换，浮点数转换
    df['num'] = pd.to_numeric(df['num'], errors='coerce')
    



# 特征获取
def get_feature():
    # 日期特征
    df['dayofyear'] = df['time'].dt.dayofyear
    df['weekofyear'] = df['time'].dt.weekofyear
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute


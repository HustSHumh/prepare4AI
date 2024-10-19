import pandas as pd
import numpy as np
import os




# 获取数据
def get_data():
    # 数据获取
    df_train_info = pd.read_csv(f'data\A榜-训练集_海上风电预测_基本信息.csv', encoding = 'gbk')
    df_train_2 = pd.read_csv(f'data\A榜-训练集_海上风电预测_基本信息.csv', encoding='gbk')
    # 数据合并    
    df_train_all = df_train_2.merge(df_train_info[['col1', 'col2']], how='left', on=['col1'])
    # 数据

    


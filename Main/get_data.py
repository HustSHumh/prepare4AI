import pandas as pd
import numpy as np
import os




# 获取数据
def get_data():
    df_train_info = pd.read_csv(f'data\A榜-训练集_海上风电预测_基本信息.csv', encoding = 'gbk')
    df_train_2 = pd.read_csv(f'data\A榜-训练集_海上风电预测_基本信息.csv', encoding='gbk')    
    df_train_all = df_train_2.merge(df_train_info[['col1', 'col2']], how='left', on=['col1'])
    







# 数据格式转换
def trans_data():
    # 转换为时间，适用于序列
    df = pd.to_datetime(df)
    # 转化为数字
    df = pd.to_numeric(df)
    # 插值
    # 平移
    # 去重
    # 保留N位小数
    # 排序
    # 去除超大值
    # dataframe 的循环

    


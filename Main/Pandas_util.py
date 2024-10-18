import pandas as pd


df = pd.read_csv('data\A榜-训练集_海上风电预测_基本信息.csv', encoding='gbk')
# 1、数据的遍历，按行遍历
for idx, raw in df.iterrows():
    print(idx, raw)

# 2、数据的信息，列
df.columns



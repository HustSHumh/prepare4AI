import matplotlib.pyplot as plt
import pandas as pd 


df = pd.DataFrame()

# 散点图
# 用于分析电厂的位置分布
plt.scatter(x=df['x'], y=df['y'], c='color', marker='o', label='1')

# 折线图
# 用于观察各曲线的变化情况
oid_name = df_station.loc[df_station['oid']==oid, 'data_name'].values[0]
df_power = df_power.loc[pred_date: pred_date, :]
acc_rate = acc_rate_all.loc[acc_rate_all['oid'] == oid , col].values[0]
cap = df_station.loc[df_station['oid']==oid, 'capacity'].values[0]
plt.figure(figsize=(16, 12))
plt.plot(df_pred_['y_pred'], color='g', label='pred')
plt.plot(df_power.loc[df_power['oid']==oid, 'power'], color='r', label='true')
plt.hlines(cap * 0.1, df_pred_.index[0],  df_pred_.index[-1], label='10%cap', color='black')
plt.title(oid+' ,capacity:{} , acc:{}, name:{}'.format(cap, acc_rate, oid_name))
plt.legend()

# 整个函数使用功能
fig, axs = plt.subplot(2, 2)
axs[0, 0].plot()
axs[0, 1].plot()


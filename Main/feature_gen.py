import pandas as pd

df = pd.DataFrame()

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
    df["季节"] = df["时间"].dt.quarter
    return 

def gen_feature():
    # 构造特征
    # 基本特征构造
    df['比大气压'] = df['气压(Pa）'] / df['温度（K）']
    #将温度转换成摄氏度
    df['温度（K）']=df['温度（K）']-273.15
    #相对湿度的取值范围为0.01~0.99
    df['相对湿度（%）'] = df['相对湿度（%）'].apply(lambda x:max(1,x))
    df['相对湿度（%）'] = df['相对湿度（%）'].apply(lambda x:min(99,x))

    #气象数据的特征工程
    #露点温度（°C） = 温度（°C） - (100 - 相对湿度（%）) / 5
    df['露点温度'] = df['温度（K）'] - (100 - df['相对湿度（%）']) / 5

    #风速和风向的特征工程
    df['sin_100m风速（100m/s）'] = df['100m风速（100m/s）'] * np.sin(np.pi * df['100m风向（°)'] / 180)
    df['cos_100m风速（100m/s）'] = df['100m风速（100m/s）'] * np.cos(np.pi * df['100m风向（°)'] / 180)

    df['sin_10米风速（10m/s）'] = df['10米风速（10m/s）'] * np.sin(np.pi * df['10米风向（°)'] / 180)
    df['cos_10米风速（10m/s）'] = df['10米风速（10m/s）'] * np.cos(np.pi * df['10米风向（°)'] / 180)
    
    df['100m风向（°)类别'] = (df['100m风向（°)'] + 1) // 90
    df['10米风向（°)类别'] = (df['10米风向（°)'] + 1) // 90
    df['10米风向（°)_100m风向（°)'] = (df['100m风向（°)类别'] == df['10米风向（°)类别'])
    
    for col in ['100m风向（°)类别','10米风向（°)类别']:
        unique_value = df[col].unique()
        for value in unique_value:
            df[col + "_" + str(value)] = (df[col] == value)

    # 滑动窗口
    gaps = [1, 2, 4, 7, 15, 30, 50, 80]
    for gap in gaps:
        for col in ['气压(Pa）', '相对湿度（%）', '云量', '10米风速（10m/s）', '10米风向（°)', '温度（K）', '辐照强度（J/m2）', '降水（m）', '100m风速（100m/s）', '100m风向（°)']:
            logging.info(f"特征{col}的{gap}gap")
            total_df[col + f"_shift{gap}"] = total_df[col].groupby(total_df['站点编号']).shift(gap)
            total_df[col + f"_gap{gap}"] = total_df[col+f"_shift{gap}"] - total_df[col]
            total_df.drop([col + f"_shift{gap}"], axis=1, inplace=True)
    

    # 特征构建
    num_cols = ['airPressure','relativeHumidity','cloudiness','10mWindSpeed','10mWindDirection',
            'temperature','irradiation','precipitation','100mWindSpeed','100mWindDirection']

    for col in tqdm.tqdm(num_cols):
        # 历史平移/差分特征
        for i in [1,2,3,4,5,6,7,15,30,50] + [1*96,2*96,3*96,4*96,5*96]:
            df[f'{col}_shift{i}'] = df.groupby('stationId')[col].shift(i)
            df[f'{col}_feture_shift{i}'] = df.groupby('stationId')[col].shift(-i)

            df[f'{col}_diff{i}'] = df[f'{col}_shift{i}'] - df[col]
            df[f'{col}_feture_diff{i}'] = df[f'{col}_feture_shift{i}'] - df[col]

            df[f'{col}_2diff{i}'] = df.groupby('stationId')[f'{col}_diff{i}'].diff(1)
            df[f'{col}_feture_2diff{i}'] = df.groupby('stationId')[f'{col}_feture_diff{i}'].diff(1)
        
        # 均值相关
        df[f'{col}_3mean'] = (df[f'{col}'] + df[f'{col}_feture_shift1'] + df[f'{col}_shift1'])/3
        df[f'{col}_5mean'] = (df[f'{col}_3mean']*3 + df[f'{col}_feture_shift2'] + df[f'{col}_shift2'])/5
        df[f'{col}_7mean'] = (df[f'{col}_5mean']*5 + df[f'{col}_feture_shift3'] + df[f'{col}_shift3'])/7
        df[f'{col}_9mean'] = (df[f'{col}_7mean']*7 + df[f'{col}_feture_shift4'] + df[f'{col}_shift4'])/9
        df[f'{col}_11mean'] = (df[f'{col}_9mean']*9 + df[f'{col}_feture_shift5'] + df[f'{col}_shift5'])/11
        
        df[f'{col}_shift_3_96_mean'] = (df[f'{col}_shift{1*96}'] + df[f'{col}_shift{2*96}'] + df[f'{col}_shift{3*96}'])/3
        df[f'{col}_shift_5_96_mean'] = (df[f'{col}_shift_3_96_mean']*3 + df[f'{col}_shift{4*96}'] + df[f'{col}_shift{5*96}'])/5
        df[f'{col}_future_shift_3_96_mean'] = (df[f'{col}_feture_shift{1*96}'] + df[f'{col}_feture_shift{2*96}'] + df[f'{col}_feture_shift{3*96}'])/3
        df[f'{col}_future_shift_5_96_mean'] = (df[f'{col}_future_shift_3_96_mean']*3 + df[f'{col}_feture_shift{4*96}'] + df[f'{col}_feture_shift{5*96}'])/3
        
        # 窗口统计
        for win in [3,5,7,14,28]:
            df[f'{col}_win{win}_mean'] = df.groupby('stationId')[col].rolling(window=win, min_periods=3, closed='left').mean().values
            df[f'{col}_win{win}_max'] = df.groupby('stationId')[col].rolling(window=win, min_periods=3, closed='left').max().values
            df[f'{col}_win{win}_min'] = df.groupby('stationId')[col].rolling(window=win, min_periods=3, closed='left').min().values
            df[f'{col}_win{win}_std'] = df.groupby('stationId')[col].rolling(window=win, min_periods=3, closed='left').std().values
            df[f'{col}_win{win}_skew'] = df.groupby('stationId')[col].rolling(window=win, min_periods=3, closed='left').skew().values
            df[f'{col}_win{win}_kurt'] = df.groupby('stationId')[col].rolling(window=win, min_periods=3, closed='left').kurt().values
            df[f'{col}_win{win}_median'] = df.groupby('stationId')[col].rolling(window=win, min_periods=3, closed='left').median().values
            
            df = df.sort_values(['stationId','time'], ascending=False)
            
            df[f'{col}_future_win{win}_mean'] = df.groupby('stationId')[col].rolling(window=win, min_periods=3, closed='left').mean().values
            df[f'{col}_future_win{win}_max'] = df.groupby('stationId')[col].rolling(window=win, min_periods=3, closed='left').max().values
            df[f'{col}_future_win{win}_min'] = df.groupby('stationId')[col].rolling(window=win, min_periods=3, closed='left').min().values
            df[f'{col}_future_win{win}_std'] = df.groupby('stationId')[col].rolling(window=win, min_periods=3, closed='left').std().values
            df[f'{col}_future_win{win}_skew'] = df.groupby('stationId')[col].rolling(window=win, min_periods=3, closed='left').skew().values
            df[f'{col}_future_win{win}_kurt'] = df.groupby('stationId')[col].rolling(window=win, min_periods=3, closed='left').kurt().values
            df[f'{col}_future_win{win}_median'] = df.groupby('stationId')[col].rolling(window=win, min_periods=3, closed='left').median().values
            
            df = df.sort_values(['stationId','time'], ascending=True)
            
            # 二阶特征
            df[f'{col}_win{win}_mean_loc_diff'] = df[col] - df[f'{col}_win{win}_mean']
            df[f'{col}_win{win}_max_loc_diff'] = df[col] - df[f'{col}_win{win}_max']
            df[f'{col}_win{win}_min_loc_diff'] = df[col] - df[f'{col}_win{win}_min']
            df[f'{col}_win{win}_median_loc_diff'] = df[col] - df[f'{col}_win{win}_median']
            
            df[f'{col}_future_win{win}_mean_loc_diff'] = df[col] - df[f'{col}_future_win{win}_mean']
            df[f'{col}_future_win{win}_max_loc_diff'] = df[col] - df[f'{col}_future_win{win}_max']
            df[f'{col}_future_win{win}_min_loc_diff'] = df[col] - df[f'{col}_future_win{win}_min']
            df[f'{col}_future_win{win}_median_loc_diff'] = df[col] - df[f'{col}_future_win{win}_median']
            
    for col in ['is_precipitation']:
        for win in [4,8,12,20,48,96]:
            df[f'{col}_win{win}_mean'] = df.groupby('stationId')[col].rolling(window=win, min_periods=3, closed='left').mean().values
            df[f'{col}_win{win}_sum'] = df.groupby('stationId')[col].rolling(window=win, min_periods=3, closed='left').sum().values
    
    
    
    
    
    # 当日辐照强度比值

    df["日期"] = df["时间"].dt.date
    day_max_values = df[["光伏用户编号", "日期", "辐照强度（J/m2）"]].groupby(by=["光伏用户编号", "日期"]).max()
    day_max_values = day_max_values.rename(columns={x: x + "_max" for x in day_max_values.columns}).reset_index()
    df = pd.merge(df, day_max_values, on=["光伏用户编号", "日期"], how="left").drop(columns=["日期"])
    df["辐照强度（J/m2）_max"] = df["辐照强度（J/m2）"] / df["辐照强度（J/m2）_max"]

    # 当日平均值
    df["日期"] = df["时间"].dt.date
    day_mean_values = df[["光伏用户编号", "日期", "是白天", "辐照强度（J/m2）"]].groupby(by=["光伏用户编号", "日期", "是白天"]).mean()
    day_mean_values = day_mean_values.rename(columns={x: x + "_mean" for x in day_mean_values.columns}).reset_index()
    df = pd.merge(df, day_mean_values, on=["光伏用户编号", "日期", "是白天"], how="left").drop(columns=["日期"])

    # 当日最高温度最低温度查
    df["日期"] = df["时间"].dt.date
    day_max_values = df[["光伏用户编号", "日期", "温度（K）"]].groupby(by=["光伏用户编号", "日期"]).max()
    day_min_values = df[["光伏用户编号", "日期", "温度（K）"]].groupby(by=["光伏用户编号", "日期"]).min()
    day_max_values = day_max_values.rename(columns={x: x + "_max" for x in day_max_values.columns}).reset_index()
    day_min_values = day_min_values.rename(columns={x: x + "_min" for x in day_min_values.columns}).reset_index()
    df = pd.merge(df, day_max_values, on=["光伏用户编号", "日期"], how="left")
    df = pd.merge(df, day_min_values, on=["光伏用户编号", "日期"], how="left").drop(columns=["日期"])
    df["温度（K）_max"] = df["温度（K）_max"] - df["温度（K）"]
    df["温度（K）_min"] = df["温度（K）"] - df["温度（K）_min"]
    df = df.rename(columns={
        "辐照强度（J/m2）_max": "光照/当天最强光照",
        "温度（K）_max": "与当天最高温度之差",
        "温度（K）_min": "与当天最低温度之差"
    })



# onehot
for col in ['站点编号']:
    logging.info(f"特征{col}的onehot")
    unique_value = total_df[col].unique()
    for value in unique_value:
        total_df[col + "_" + str(value)] = (total_df[col] == value)
    total_df.drop([col], axis=1, inplace=True)






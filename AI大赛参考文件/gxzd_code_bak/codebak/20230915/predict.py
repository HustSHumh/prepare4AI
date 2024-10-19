import os
import gc
import numpy as np
import pandas as pd
from odps import ODPS
from odps.df import DataFrame
import datetime
import time
import matplotlib.pyplot as plt
import openpyxl
import catboost as cb
from tqdm import tqdm
from pvlib import location
import warnings
import pickle
warnings.filterwarnings('ignore')

from sklearn.neighbors import KDTree


# 公共OSS AK
bucket_name = 'sf-2023'
oss_endpoint = 'http://oss-cn-guangzhou-nfdw-d01-a.pdcc-cloud-inc.cn'
AK = 'wbr4Y7IjXopW7CWZ'
AKS = 'reP1eCeahBC7U9w0i5rhQU9oGgS0NA'

# 广西AK
AK_GX = 'MBs3yHviVlKIY3gh'
AKS_GX = 'wUvWZOnGTtWVVir65A3ogIfOKSkHIB'
NAME = 'sf_2023_chenquanqi'
endpoint = 'http://service.cn-guangzhou-nfdw-d01.odps.pdcc-cloud-inc.cn/api'


oid_list = ['15481140756807681','15481140756873217','15481140756938753','15481140830863361','15481140766113793',
            '15481140766179329','15481140766244865','15481140766310401','15481140757266433','15481140757725185',
            '15481140757790721','15481140757004289','15481140757069825','15481140757135361','15481140757200897',
            '15481140766375937','15481140766441473','15481140766507009','15481140766572545','15481137326915585',
            '15481130476634115','15481130500882435','15481125255184390','15481129194225670','15481128900034564',
            '15481128891252742','15481130821025794','15481130821156866','15481130901110786','15481131802034178']

def get_power_station():
    df = pd.read_pickle('~/workspace/data/station/20230821.p')
    return df

def get_oid_capacity():
    df = pd.read_pickle('~/workspace/data/oid/newest.p')
    return df

def get_power_data(start_date, end_date, oid_list):
    df_ls = []
    for date in tqdm(pd.date_range(start_date, end_date, freq='D')):
        df_date = pd.read_pickle('~/workspace/data/power/{}.p'.format(str(date.date())))
        df_date = df_date[df_date['oid'].isin(oid_list)]
        df_date = df_date[['power_time', 'oid', 'type', 'power']]
        df_date.drop_duplicates(subset=['power_time', 'oid', 'type'], keep='last', inplace=True)
        df_date.set_index('power_time', inplace=True)
        df_date['oid'] = df_date['oid'].astype(str)
        df_date['type'] = df_date['type'].astype(str)
        df_date['power'] = df_date['power'].astype(float).apply(lambda x: round(x, 8))
        df_date = df_date.tz_localize('Asia/Shanghai').sort_index()
        df_ls.append(df_date)
    df = pd.concat(df_ls)



def prepare_power(start_date, end_date, oid_list):
    df_ls = []
    for date in tqdm(pd.date_range(start_date, end_date, freq='D')):
        df_date = pd.read_pickle('~/workspace/data/power/{}.p'.format(str(date.date())))
        df_date = df_date[df_date['oid'].isin(oid_list)]
        df_date = df_date[['power_time', 'oid', 'type', 'power']]
        df_date.drop_duplicates(subset=['power_time', 'oid', 'type'], keep='last', inplace=True)
        df_date.set_index('power_time', inplace=True)
        df_date['oid'] = df_date['oid'].astype(str)
        df_date['type'] = df_date['type'].astype(str)
        df_date['power'] = df_date['power'].astype(float).apply(lambda x: round(x, 8))
        df_date = df_date.tz_localize('Asia/Shanghai').sort_index()
        df_ls.append(df_date)
    df = pd.concat(df_ls)
    return df

def get_jiutian2qxj_data(start_date, end_date, flag):
    df_ls = []
    for date in tqdm(pd.date_range(start_date, end_date, freq='D')):
        df_date = pd.read_pickle('~/workspace/data/jiutian2qxj/{}.p'.format(str(date.date())+ flag))
        df_ls.append(df_date)
    df = pd.concat(df_ls)
    df = df.sort_index()
    return df    



def get_clearsky(dt_idx, lat, lon, alt=0):
    site = location.Location(lat, lon, altitude=alt, tz='Asia/Shanghai')
    df_clearsky = site.get_clearsky(dt_idx, model='ineichen')
    return df_clearsky



def create_loc_feature(df_jiutian2qxj_data, feature_list):
    align_pred_ls = []
    df_jiutian2qxj_data['loc'] = df_jiutian2qxj_data[['jd', 'wd']].apply(tuple, axis=1)
    for loc in df_jiutian2qxj_data['loc'].unique():
        df_pred_ = pd.DataFrame(index=df_jiutian2qxj_data.index.unique())
        df_pred_ = df_jiutian2qxj_data.loc[df_jiutian2qxj_data['loc'] == loc, feature_list]
        df_pred_.columns = ['{}_{}_{}'.format(feature, loc[0], loc[1]) for feature in feature_list]
        align_pred_ls.append(df_pred_)
    df_pred_aligned = pd.concat(align_pred_ls, axis=1).interpolate(limit=3)
    return df_pred_aligned.applymap(lambda x: x if x >= 0 else 0)

def post_process_oid(df_pred):
    df_station = get_oid_capacity()
    for oid,df_pred_ in df_pred.groupby('oid'):
        station_r1 = df_station.loc[df_station['oid'] == oid,'capacity'].values[0] * 0.1
        df_pred.loc[df_pred['oid'] == oid,'y_pred'] = df_pred_['y_pred'].apply(lambda x: x if x >= station_r1 else station_r1)
    return df_pred

def create_dir_if_not_exist(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return dirpath

def send_results(df_pred):
    df_res = pd.DataFrame()
    df_res['oid'] = df_pred['oid'].values
    df_res['sjrq'] = [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * 2880
    df_res['ycrq'] = df_pred.index.tz_localize(None).to_list()
    df_res['ycz'] = df_pred['y_pred'].values
    df_res['rksj'] = [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * 2880
    df_res['dwmc'] = ['sf_2023_chenquanqi'] * 2880 
    o = ODPS(
        AK_GX,
        AKS_GX,
        NAME,
        endpoint
        )
    # 入库
    DataFrame(df_res).persist('t_power_forecast', odps=o)
    return 

def predict():

    pred_date = datetime.date.today() + datetime.timedelta(days=1)
    pred_today = pred_date - datetime.timedelta(days=1)
    his_start_date = '2023-08-02'
    his_end_date = str(pred_date - datetime.timedelta(days=1))
    res_dirpath = 'results_v1/{}'.format(pred_date)
    create_dir_if_not_exist(res_dirpath)

    # 获取8月2日至今的数据
    df_power_data = get_power_data(his_start_date, his_end_date, oid_list)
#     print('完成获取OID历史功率数据，共耗时 ', datetime.datetime.now() - now_time, 's')
    # df_qxj_weather_data = get_qxj_weather_data(his_end_date, his_end_date)
    #     print('完成获取气象局风电气象数据，共耗时 ', datetime.datetime.now() - now_time, 's')
    # df_qxj_radiation_data = get_qxj_radi_data(his_end_date, his_end_date)
    df_jiutian2qxj_wind_data = get_jiutian2qxj_data(his_start_date, pred_date, 'wind')
    df_jiutian2qxj_radi_data = get_jiutian2qxj_data(his_start_date, pred_date, 'radi')
#     print('完成获取玖天数据，共耗时 ', datetime.datetime.now() - now_time, 's')
    df_power = pd.read_pickle(os.path.join(res_dirpath, 'df_power.p'))

    df_wind_align = create_loc_feature(df_jiutian2qxj_wind_data, 
                                       ['ten_meter_wind_speed', 'one_hundred_wind_speed', 'ten_meter_wind_direction', 'one_hundred_meter_wind_direction'])
    df_wind_align.to_pickle(os.path.join(res_dirpath, 'df_wind_align.p'))
#     print('完成风力特征训练，共耗时 ', datetime.datetime.now() - now_time, 's')
    df_radi_align = create_loc_feature(df_jiutian2qxj_radi_data, ['total_radiation'])
    df_radi_align.to_pickle(os.path.join(res_dirpath, 'df_radi_align.p'))
#     print('完成光伏特征训练，共耗时 ', datetime.datetime.now() - now_time, 's')

    power_train_start = '2023-08-02' + ' 00:00'
    power_train_end = str(pred_today - datetime.timedelta(days=1)) + ' 23:45'
    # power_val_start = str(datetime.date.today() - datetime.timedelta(days=6)) + ' 00:00'
    # power_val_end = str(datetime.date.today() - datetime.timedelta(days=1)) + ' 23:45'
    power_pred_start = str(pred_today + datetime.timedelta(days=1)) + ' 00:00'
    power_pred_end = str(pred_today + datetime.timedelta(days=1)) + ' 23:45'

    train_data_ls = []
    # val_data_ls = []
    pred_data_ls = []
    for oid, df_oid in tqdm(df_power.groupby('oid')):
        data_ = pd.concat([df_oid, df_wind_align, df_radi_align], axis=1)
        data_['oid'] = data_['oid'].ffill()
        data_['type'] = data_['type'].ffill()
        data_['oid'] = data_['oid'].astype(str)
        data_['type'] = data_['type'].astype(str)
        data_['hour'] = data_.index.hour
        data_['minute'] = data_.index.minute
        data_['hour_sin'] = np.sin(data_['hour'] / 23 * 2 * np.pi)
        data_['hour_cos'] = np.cos(data_['hour'] / 23 * 2 * np.pi)
        data_['minute_sin'] = np.sin(data_['minute'] / 59 * 2 * np.pi)
        data_['minute_cos'] = np.cos(data_['minute'] / 59 * 2 * np.pi)
        data_['time_96'] = data_.apply(lambda x: (x['hour'] * 60 + x['minute']) / 15 + 1, axis=1)
        data_['hour'] = data_['hour'].astype(str)
        data_['minute'] = data_['minute'].astype(str)
        data_['num_samples'] = list(range(len(data_)))
        train_data_ = data_.loc[power_train_start:power_train_end]
    #     val_data_ = data_.loc[power_val_start:power_val_end]
        pred_data_ = data_.loc[power_pred_start:power_pred_end]
        train_data_ls.append(train_data_)
    #     val_data_ls.append(val_data_)
        pred_data_ls.append(pred_data_)
    train_data = pd.concat(train_data_ls)
    # val_data = pd.concat(val_data_ls)
    pred_data = pd.concat(pred_data_ls)
    X_train, y_train = train_data.loc[:, [i for i in train_data if i != 'power']], train_data['power']
    # X_val, y_val = val_data.loc[:, [i for i in val_data if i != 'power']], val_data['power']
    X_pred = pred_data.loc[:, [i for i in pred_data if i != 'power']]
    train_nan_mask = y_train.isna().values
    X_train = X_train[~train_nan_mask]
    y_train = y_train[~train_nan_mask]
    print('Start training...')
    num_iterations = 3000
    model = cb.CatBoostRegressor(iterations=num_iterations, loss_function='MAE', random_state=0, verbose=True, cat_features=['oid', 'type', 'hour', 'minute'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_pred)
    df_pred = X_pred.loc[:, ['oid', 'type']].copy()
    df_pred['y_pred'] = y_pred
    #post process
    df_pred = post_process_oid(df_pred)
    df_pred.to_pickle(os.path.join(res_dirpath, 'df_pred.p'))
    return df_pred


def main():
    df_pred = predict()
    send_results(df_pred)
    return


if __name__ == '__main__':
    print('程序开始运行')
    while True:
        now = datetime.datetime.now()
        if (now.hour == 15 and now.minute == 30):
            print('开始预测明日结果')
            main()
        time.sleep(55)
        print('预测脚本运行中, beat, ', now)





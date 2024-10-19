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

def pull_jiutian_wind(ybsj1, ybsj2):
    o = ODPS(
        AK_GX,
        AKS_GX,
        NAME,
        endpoint
    )
    run_start_time = datetime.datetime.now()
    
    start_time = str(ybsj1)+ ' 00:00:00'
    end_time = str(ybsj2) + ' 23:45:00'
    print('开始读取玖天风向风速数据', datetime.datetime.now() - run_start_time)
    sql = '''select fbsj, ybsj, latitude, longitude, ten_meter_wind_speed, one_hundred_wind_speed, ten_meter_wind_direction, one_hundred_meter_wind_direction from sf_2023_pw.t_jiutian_gridding_data t 
    where t.fbsj >= "{}" and t.fbsj <= "{}"'''.format(start_time, end_time)
    querry_job = o.execute_sql(sql)
    result = querry_job.open_reader()
    df = result.to_pandas()
    print('读取玖天风向风速数据', datetime.datetime.now() - run_start_time)
    return df


def pull_jiutian_radiation(ybsj1, ybsj2):
    o = ODPS(
        AK_GX,
        AKS_GX,
        NAME,
        endpoint
    )
    run_start_time = datetime.datetime.now()
    
    start_time = str(ybsj1)+ ' 00:00:00'
    end_time = str(ybsj2) + ' 23:45:00'
    sql = '''select fbsj, ybsj, latitude, longitude, total_radiation from sf_2023_pw.t_jiutian_gridding_data t 
    where t.fbsj >= "{}" and t.fbsj <= "{}"'''.format(start_time, end_time)
    querry_job = o.execute_sql(sql)
    result = querry_job.open_reader()
    df = result.to_pandas()
    print('读取玖天辐照度数据', datetime.datetime.now() - run_start_time)
    return df


def pull_qxj_weather_data(table_name, ybsj1, ybsj2):
    o = ODPS(
        AK_GX,
        AKS_GX,
        NAME,
        endpoint
    )
    
    run_start_time = datetime.datetime.now()
    
    start_time = str(ybsj1) + ' 00:00:00'
    end_time = str(ybsj2) + ' 23:45:00'
    sql = '''select * from sf_2023_pw.{} t 
    where t.zlsc >= "{}" and t.zlsc <= "{}"'''.format(str(table_name), str(start_time), end_time)
    querry_job = o.execute_sql(sql)
    result = querry_job.open_reader()
    df = result.to_pandas()
    df.sort_values(by=['zlsc', 'qxjzh'], inplace=True)
    print('读取气象局风电气象数据', datetime.datetime.now() - run_start_time)
    return df


def pull_qxj_radiation_data(table_name, ybsj1, ybsj2):
    o = ODPS(
        AK_GX,
        AKS_GX,
        NAME,
        endpoint
    )
    
    run_start_time = datetime.datetime.now()
    
    start_time = str(ybsj1) + ' 00:00:00'
    end_time = str(ybsj2) + ' 23:45:00'
    sql = '''select * from sf_2023_pw.{} t 
    where t.zlsc >= "{}" and t.zlsc <= "{}"'''.format(str(table_name), str(start_time), end_time)
    querry_job = o.execute_sql(sql)
    result = querry_job.open_reader()
    df = result.to_pandas()
    df.sort_values(by=['zlsc', 'qxjzh'], inplace=True)
    print('读取气象局光伏气象数据', datetime.datetime.now() - run_start_time)
    return df


def pull_power_data(start_date, end_date):
    # update_all = True 时全量更新数据
    o = ODPS(
            AK_GX,
            AKS_GX,
            NAME,
            endpoint
        )
    run_start_time = datetime.datetime.now()
    start_time = str(start_date)+ ' 00:00:00'
    end_time = str(end_date) + ' 23:45:00'

    sql = '''select * from sf_2023_pw.t_electricity t 
    where t.power_time >= "{}" and t.power_time <= "{}"'''.format(start_time, end_time)
    querry_job = o.execute_sql(sql)
    result = querry_job.open_reader()
    df = result.to_pandas()
    print('读取发电功率数据', datetime.datetime.now() - run_start_time)
    return df

def update_capacity():
    o = ODPS(
        AK_GX,
        AKS_GX,
        NAME,
        endpoint
    )
    sql = '''select oid, data_name, acc_month, capacity from sf_2023_pw.td_capacity t '''
    querry_job = o.execute_sql(sql)
    result = querry_job.open_reader()
    df = result.to_pandas()
    df.sort_values('acc_month', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['oid'] = df['oid'].astype(str)
    df['capacity'] = df['capacity'].astype(float)

    return df


def create_dir_if_not_exist(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return dirpath



def prepare_power(start_date, end_date, oid_list):
    df_ls = []
    for date in tqdm(pd.date_range(start_date, end_date, freq='D')):
        df_date = pd.read_pickle('data/power/{}.p'.format(str(date.date())))
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


def get_qxj_weather_data(start_date, end_date):
    df_ls = []
    for date in tqdm(pd.date_range(start_date, end_date, freq='D')):
        df_date = pd.read_pickle('data/qxj_weather/{}.p'.format(str(date.date())))
        df_date = df_date[['zlsc', 'wd', 'jd', 'hbgd', 'fx2fz', 'pjfs2fz', 'fx10fz', 'pjfs10fz']]
        df_date.drop_duplicates(subset=['zlsc', 'wd', 'jd', 'hbgd'], keep='last', inplace=True)
        df_date.set_index('zlsc', inplace=True)
        df_date = df_date.astype(float).apply(lambda x: round(x, 8))
        df_date['fx2fz'] = df_date['fx2fz'].apply(lambda x: x if x < 361 else np.nan).interpolate(limit=1)
        df_date['fx10fz'] = df_date['fx10fz'].apply(lambda x: x if x < 361 else np.nan).interpolate(limit=1)
        df_date['pjfs2fz'] = df_date['pjfs2fz'].apply(lambda x: x if x < 99 else np.nan).interpolate(limit=1)
        df_date['pjfs10fz'] = df_date['pjfs10fz'].apply(lambda x: x if x < 99 else np.nan).interpolate(limit=1)
        df_ls.append(df_date)
    df = pd.concat(df_ls)
    df = df.tz_localize('Asia/Shanghai').sort_index()
    return df


def get_qxj_radi_data(start_date, end_date):
    df_ls = []
    for date in tqdm(pd.date_range(start_date, end_date, freq='D')):
        df_date = pd.read_pickle('data/qxj_radiation/{}.p'.format(str(date.date())))
        df_date = df_date[['zlsc', 'wd', 'jd', 'hbgd', 'zfzd']]
        df_date.drop_duplicates(subset=['zlsc', 'wd', 'jd', 'hbgd'], keep='last', inplace=True)
        df_date.set_index('zlsc', inplace=True)
        df_date = df_date.astype(float).apply(lambda x: round(x, 8))
        df_date['zfzd'] = df_date['zfzd'].apply(lambda x: x if x < 2999 else np.nan).interpolate(limit=1)
        df_ls.append(df_date)
    df = pd.concat(df_ls)
    df = df.tz_localize('Asia/Shanghai').sort_index()
    return df


def get_jiutian_wind_data(start_date, end_date):
    df_ls = []
    for date in tqdm(pd.date_range(start_date, end_date, freq='D')):
        df_date = pd.read_pickle('data/jiutian_wind/{}.p'.format(str(date.date())))
        mask = ((df_date['ybsj'] - df_date['fbsj']) > datetime.timedelta(hours=21, minutes=45)) & \
        ((df_date['ybsj'] - df_date['fbsj']) < datetime.timedelta(hours=46))
        df_date = df_date[mask]
        df_date.drop_duplicates(subset=['ybsj', 'latitude', 'longitude'], keep='last', inplace=True)
        df_date.drop('fbsj', axis=1, inplace=True)
        df_date.set_index('ybsj', inplace=True)
        df_date = df_date.astype(float).apply(lambda x: round(x, 8))
        df_date['ten_meter_wind_speed'] = df_date['ten_meter_wind_speed'].apply(lambda x: x if x < 99 else np.nan).interpolate(limit=1)
        df_date['one_hundred_wind_speed'] = df_date['one_hundred_wind_speed'].apply(lambda x: x if x < 99 else np.nan).interpolate(limit=1)
        df_date['ten_meter_wind_direction'] = df_date['ten_meter_wind_direction'].apply(lambda x: x if x < 361 else np.nan).interpolate(limit=1)
        df_date['one_hundred_meter_wind_direction'] = df_date['one_hundred_meter_wind_direction'].apply(lambda x: x if x < 361 else np.nan).interpolate(limit=1)
        df_ls.append(df_date)
    df = pd.concat(df_ls)
    df = df.tz_localize('Asia/Shanghai').sort_index()
    return df

def get_jiutian_radi_data(start_date, end_date):
    df_ls = []
    for date in tqdm(pd.date_range(start_date, end_date, freq='D')):
        df_date = pd.read_pickle('data/jiutian_radiation/{}.p'.format(str(date.date())))
        mask = ((df_date['ybsj'] - df_date['fbsj']) > datetime.timedelta(hours=21, minutes=45)) & \
        ((df_date['ybsj'] - df_date['fbsj']) < datetime.timedelta(hours=46))
        df_date = df_date[mask]
        df_date.drop_duplicates(subset=['ybsj', 'latitude', 'longitude'], keep='last', inplace=True)
        df_date.drop('fbsj', axis=1, inplace=True)
        df_date.set_index('ybsj', inplace=True)
        df_date = df_date.astype(float).apply(lambda x: round(x, 8))
        df_date['total_radiation'].astype(float).apply(lambda x: round(x, 8))
        df_date['total_radiation'] = df_date['total_radiation'].apply(lambda x: x if x < 2999 else np.nan).interpolate(limit=1)
        df_ls.append(df_date)
    df = pd.concat(df_ls)
    df = df.tz_localize('Asia/Shanghai').sort_index()
    return df





def filter_jiutian_station(df_points_1, df_points_2, k, flag='jiutian'):
    start_time = datetime.datetime.now()
    df_points_1.reset_index(drop=False, inplace=True)
    df_points_2.reset_index(drop=False, inplace=True)
    power_time = 'zlsc'
    jd = 'jd'
    wd = 'wd'
    if flag == 'jiutian':
        jd = 'longitude'
        wd = 'latitude'
        power_time = 'ybsj'
    df_loc_1 = df_points_1.loc[:, ['jd', 'wd']].drop_duplicates() # dataframe   (n1, 2)
    df_loc_2 = df_points_2.loc[:, [jd, wd]].drop_duplicates()  # dataframe (n_station, 2)
    df_loc_2.reset_index(drop=True, inplace=True)
    df_loc_1.reset_index(drop=True, inplace=True)
    # KD树只返回相对index
    n_time = df_points_2[power_time].unique().shape[0]
#     print('开始建树', datetime.datetime.now() - start_time)
    Tree = KDTree(df_loc_2)
    dist, ind = Tree.query(df_loc_1, k) # ind: n1*5
#     print('建树完成', datetime.datetime.now() - start_time)
    df_jiutian_dict = {}
    for idx in ind:
        jd_tmp_list = df_loc_2.loc[idx, jd].tolist()
        wd_tmp_list = df_loc_2.loc[idx, wd].tolist()
        for jd_, wd_ in zip(jd_tmp_list, wd_tmp_list): 
            df_jiutian_dict[(jd_, wd_)] = df_jiutian_dict.get((jd_, wd_), 0) + 1
    return df_jiutian_dict


def simulate_curve3(df_points_1, df_points_2, k, feature_list, flag='jiutian'):
    df_points_2.reset_index(drop=False, inplace=True)
    start_time = datetime.datetime.now()
    power_time = 'zlsc'
    jd = 'jd'
    wd = 'wd'
    if flag == 'jiutian':
        jd = 'longitude'
        wd = 'latitude'
        power_time = 'ybsj'
    df_loc_1 = df_points_1.loc[:, ['jd', 'wd']].drop_duplicates() # dataframe   (n1, 2)
    df_loc_2 = df_points_2.loc[:, [jd, wd]].drop_duplicates()  # dataframe (n_station, 2)
    df_loc_2.reset_index(drop=True, inplace=True)
    df_loc_1.reset_index(drop=True, inplace=True)
    if power_time not in feature_list:
        feature_list.append(power_time)
    # KD树只返回相对index
    n_time = df_points_2[power_time].unique().shape[0]
    Tree = KDTree(df_loc_2)
    dist, ind = Tree.query(df_loc_1, k) # ind: n1*5
#     convert dataframe to dict
    df_points_dict2 = {}
    for idx, val in df_loc_2.iterrows():
        key = (val[jd], val[wd])
        data = df_points_2.loc[df_points_2['loc']==key, feature_list]
        data.drop_duplicates(subset=[power_time], inplace=True)
        df_points_dict2[key] = data
    
    station_list = []
    for idx in ind: # iterate n1 idx.shape = (5,)
        jd_tmp_list = df_loc_2.loc[idx, jd].tolist()
        wd_tmp_list = df_loc_2.loc[idx, wd].tolist()
        convert_tmp = None
        for jd_, wd_ in zip(jd_tmp_list, wd_tmp_list): #iterate k
            if convert_tmp is None:
                convert_tmp = (df_points_dict2[(jd_, wd_)]).set_index(power_time)
            else:
                convert_tmp += (df_points_dict2[(jd_, wd_)]).set_index(power_time)
        convert_tmp /= k
        station_list.append(convert_tmp)
    df_final = pd.DataFrame()
    for idx, row in df_loc_1.iterrows():
        df_tmp = pd.DataFrame(
            {
                'power_time' : df_points_2[power_time].unique(),
                'jd' : [row['jd']] * n_time,
                'wd' : [row['wd']] * n_time,
            }
        )
        df_tmp = df_tmp.set_index('power_time')
        df_tmp = pd.concat([df_tmp, station_list[idx]], axis=1)
        df_final = pd.concat([df_final, df_tmp], axis = 0) 
    return df_final


def prepare_jiutian2qxj_data():
    today = datetime.date.today()
    his_end_date = str(today)
    pred_date = str(today + datetime.timedelta(days=1))
    save_dirpath = 'data/jiutian2qxj'
    qxj_weather_feature = ['fx2fz', 'pjfs2fz', 'fx10fz', 'pjfs10fz']
    qxj_radiation_feature = ['zfzd']
    jiutian_wind_feature  = ['ten_meter_wind_speed', 'one_hundred_wind_speed', 'ten_meter_wind_direction', 'one_hundred_meter_wind_direction']
    jiutian_radi_feature = ['total_radiation']
    
    df_qxj_weather_data = get_qxj_weather_data(his_end_date, his_end_date)
#     print('完成获取气象局风电气象数据，共耗时 ', datetime.datetime.now() - now_time, 's')
    df_qxj_radiation_data = get_qxj_radi_data(his_end_date, his_end_date)
#     print('完成获取气象局光伏气象数据，共耗时 ', datetime.datetime.now() - now_time, 's')
    
    df_jiutian_wind_data = get_jiutian_wind_data(his_end_date, his_end_date)
    df_jiutian_wind_data['loc'] = df_jiutian_wind_data[['longitude', 'latitude']].apply(tuple, axis=1)
    df_jiutian_radi_data = get_jiutian_radi_data(his_end_date, his_end_date)
    df_jiutian_radi_data['loc'] = df_jiutian_radi_data[['longitude', 'latitude']].apply(tuple, axis=1)
    
    df_jiutian_filter = get_jiutian_wind_data(his_end_date, his_end_date)
#     print('完成获取玖天预测气象数据，共耗时 ', datetime.datetime.now() - now_time, 's')
    
    # 筛选出需要用到的玖天气象站
    dict_jiutian_wind = filter_jiutian_station(df_qxj_weather_data, df_jiutian_filter, 5)
    dict_jiutian_radi = filter_jiutian_station(df_qxj_radiation_data, df_jiutian_filter, 5)
    #KD tree拟合出气象局的数据
    df_jiutian_wind_data = df_jiutian_wind_data.loc[df_jiutian_wind_data['loc'].isin(dict_jiutian_wind)]
    df_jiutian_radi_data = df_jiutian_radi_data.loc[df_jiutian_radi_data['loc'].isin(dict_jiutian_radi)]


    jiutian2qxj_wind = simulate_curve3(df_qxj_weather_data, df_jiutian_wind_data,5, jiutian_wind_feature)
    jiutian2qxj_radi = simulate_curve3(df_qxj_radiation_data, df_jiutian_radi_data,5, jiutian_radi_feature)
    
    jiutian2qxj_wind.to_pickle(os.path.join(save_dirpath, '{}wind.p'.format(str(today+datetime.timedelta(days=1)))))
    jiutian2qxj_radi.to_pickle(os.path.join(save_dirpath, '{}radi.p'.format(str(today+datetime.timedelta(days=1)))))
    
    return jiutian2qxj_wind,jiutian2qxj_radi


def get_all_data():
    current_date = str(datetime.date.today())
    prev_date = str(datetime.date.today() - datetime.timedelta(days=1))

    _df = pull_jiutian_wind(current_date, current_date)
    _df.to_pickle('data/jiutian_wind/{}.p'.format(current_date))
    del _df

    _df = pull_jiutian_radiation(current_date, current_date)
    _df.to_pickle('data/jiutian_radiation/{}.p'.format(current_date))
    del _df

    _df = pull_qxj_weather_data('t_qxj_country_weather2023', current_date, current_date)
    _df.to_pickle('data/qxj_weather/{}.p'.format(current_date))
    del _df

    _df = pull_qxj_weather_data('t_qxj_country_weather2023', prev_date, prev_date)
    _df.to_pickle('data/qxj_weather/{}.p'.format(prev_date))
    del _df

    _df = pull_qxj_radiation_data('t_qxj_country_radiation', current_date, current_date)
    _df.to_pickle('data/qxj_radiation/{}.p'.format(current_date))
    del _df

    _df = pull_qxj_radiation_data('t_qxj_country_radiation', prev_date, prev_date)
    _df.to_pickle('data/qxj_radiation/{}.p'.format(prev_date))
    del _df

    _df = pull_power_data(current_date, current_date)
    _df.to_pickle('data/power/{}.p'.format(current_date))
    del _df

    _df = pull_power_data(prev_date, prev_date)
    _df.to_pickle('data/power/{}.p'.format(prev_date))
    del _df
    
    _df = update_capacity()
    _df.to_pickle('data/oid/newest.p')
    _df.to_pickle('data/oid/{}.p'.format(prev_date))
    del _df
    
    gc.collect()
    return 

def prepare_data():
    res_dirpath = '~/workspace/results_v1/{}'.format(str(datetime.date.today() + datetime.timedelta(days=1)))
    create_dir_if_not_exist(res_dirpath)
    
    power_start_date = str(datetime.date.today() - datetime.timedelta(days=120))
    power_end_date = str(datetime.date.today())
    df_power = prepare_power(power_start_date, power_end_date, oid_list)
    df_power.to_pickle(os.path.join(res_dirpath, 'df_power.p'))
    
    jiutian2qxj_wind,jiutian2qxj_radi = prepare_jiutian2qxj_data()
    
    return 
    
def main():
    get_all_data()
    prepare_data()
    return 


if __name__ == '__main__':
    print('程序开始运行')
    while True:
        now = datetime.datetime.now()
        if (now.hour == 15 and now.minute == 13):
            print('开始获取数据')
            main()
        time.sleep(55)
        print('数据获取运行中, beat, ', now)





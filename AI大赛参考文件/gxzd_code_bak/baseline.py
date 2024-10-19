import gc
import datetime
import oss2
import numpy as np
import pandas as pd
from odps import ODPS
from odps.df import DataFrame
import datetime
import time
import matplotlib.pyplot as plt
import openpyxl
import os


from tqdm import tqdm



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

def pull_yes_data(oid_list, o, date_list=None):
    date_type = '%Y-%m-%d'
    today = datetime.date.today().strftime('%Y-%m-%d')
    start_date = (datetime.date.today() +  datetime.timedelta(days=-30)).strftime('%Y-%m-%d')
    
    if date_list is None:
        date_list = [(datetime.date.today() + datetime.timedelta(days=-1)).strftime('%Y-%m-%d')] * len(oid_list)
    else:
        if len(date_list) != len(oid_list):
            raise ValueError('请输入与OID数量一致的日期数量')
        else:
            for date in date_list:
                try:
                    time.strptime(date, date_type)
                    
                    # continue
                except:
                    raise ValueError('请输入正确的日期格式——例如‘2023-01-01’')
                if date < start_date or date >= today: 
                    raise ValueError('请输入一个月内的日期， 在{}到{}之间'.format(start_date, (datetime.date.today() + datetime.timedelta(days=-1)).strftime('%Y-%m-%d')))
    
    df_list = []
    for oid, date in zip(oid_list, date_list):
        sql = '''select oid, power_time, power from sf_2023_pw.t_electricity t 
        where t.oid = {}  and t.power_time >= "{}"'''.format(oid, date + ' 00:00:00')
        querry_job = o.execute_sql(sql)
        result = querry_job.open_reader()
        df = result.to_pandas()
        df.sort_values('power_time', inplace=True)
        df['power_time'] = pd.to_datetime(df['power_time'])
        df['power'] = pd.to_numeric(df['power'])
        df = df.set_index('power_time')
        df = df.loc[date:date, :]
        df_list.append(df)
    return df_list

def send_result(df_list, o):
    pred_date = (datetime.date.today() + datetime.timedelta(days = 1)).strftime('%Y-%m-%d')
    sjrq = datetime.datetime.now().strftime('%Y-%M-%d hh:mm:ss')
    df_final = None
    for df in df_list:
        df_res = pd.DataFrame(
            {    
                'oid' : df['oid'].values,
                'sjrq' : [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * 96,
                'ycrq' : pd.date_range(pred_date, periods=96, freq='15Min'),
                'ycz' : df['power'].values,
                'rksj' : [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * 96,
                'dwmc' : ['sf_2023_chenquanqi'] * 96
            }
        )
        if df_final is None:
            df_final = df_res
        else:
            df_final = pd.concat([df_final, df_res], axis=0)
    DataFrame(df_final).persist('t_power_forecast', odps=o, overwrite=False)
    return df_final


def baseline(date_list=None):
    
    o = ODPS(
        AK_GX,
        AKS_GX,
        NAME,
        endpoint
        )
    try:
        df_list = pull_yes_data(oid_list, o, date_list)
    except:
        print('获取数据失败，请检查数据更新情况')
        return False
    try:
        df_final = send_result(df_list, o)
    except:
        print('发送数据失败，请检查发送程序')
        return False
    del df_final
    gc.collect()
    return True

def create_dir_if_not_exist(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return dirpath

if __name__ == '__main__':
    print('程序开始运行')
    sucess_date = datetime.date.today()
    while True:
        now = datetime.datetime.now()
        now_date = now.date()
        if now.hour >= 8 and now.hour <= 11 and now.minute == 0:
            if now_date != sucess_date:
                flag = baseline()
                if flag == True:
                    sucess_date = now_date
                    print(now_date, '，保底代码运行成功')
                else:
                    print(now_date, '，保底代码运行失败')
        time.sleep(55)
        print('保底脚本运行中, beat, ', now)

    
    
    
  

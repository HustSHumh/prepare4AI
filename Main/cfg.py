import os
import random


# 全局

random_seed = 2024
abs_workspace = os.path.dirname(os.path.abspath(__file__))



# Catboost

cb_params = {'iterations' : 3000,
            'random_seed' : 2024,
            'verbose' : True,
            'eval_metric' : "RMSE"}


# lighGBM

lgb_params ={
            #'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            #'min_child_weight': 5,
            #'num_leaves': 2 ** 8,
            #'lambda_l2': 10,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 4,
            #'learning_rate': 0.1,
            'seed': 2024,
            #'nthread' : 16,
            'verbose' : -1,
            }

lgb_params1 = {
    'objective': 'regression',
    'metric': 'rmse',
    'subsample' : 0.8,
    'colsample_bytree' : 0.8,
    'subsample_freq' : 5
}


# Xgboost
xgb_params = {
    'learning_rate' : 0.1,
    'n_estimators':150,
    'max_depth':5,
    'min_child_weight':1,
    'gamma':0,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'objective':'regression',
    'seed':2024
}


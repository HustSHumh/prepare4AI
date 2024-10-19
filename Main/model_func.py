import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from cfg import cb_params, lgb_params,lgb_params1, xgb_params

from sklearn.model_selection import KFold, GridSearchCV, train_test_split



df = pd.DataFrame()
# 网格寻参
valid = df.sample(frac=0.2, random_state=2024)
df.drop(index=valid.index, axis=1, inplace=True)

X = df['feature']
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)


model_lgb = lgb.LGBMRegressor(**lgb_params1)


lgb_param_test = {
    'max_depth' : range(7, 11, 1),
    'num_leaves' : range(10, 90, 10),
    'lambda_l2': [5, 10, 15],
    'min_child_weight': [1, 5, 9],
    'learning_rate' : [0.1, 0.05, 0.01],
}


gsearch1 = GridSearchCV(estimator=model_lgb, 
                        param_grid=lgb_param_test,
                        scoring='neg_root_mean_squared_error',
                        cv=5,
                        verbose=1,
                        n_jobs=-1)
gsearch1.fit(X, y)
print(gsearch1.best_params_)


model_cb = cb.CatBoostRegressor(**cb_params)

cb_params_test = {
    'iterations' : [1000, 2000, 3000],
    'learning_rate' : [0.01, 0.05, 0.1],
    'depth' : [4, 6, 8, 10],
    'l2_leaf_reg' : [1e-3, 1e-2, 1, 3]
}

gsearch2 = GridSearchCV(estimator=model_cb, 
                        param_grid=cb_params_test,
                        scoring='neg_root_mean_squared_error',
                        cv=5,
                        verbose=1,
                        n_jobs=-1)
gsearch1.fit(X, y)
print(gsearch2.best_params_)


model_xgb = xgb.XGBRegressor(**xgb_params)
xgb_params_test = {
    'n_estimators': [100, 150, 200],
    'min_child_weight' : [0.1, 1, 3],
    'max_depth' : [4,6,8],
    'learning_rate' : [0.1, 0.05, 0.01]
}
gsearch3 = GridSearchCV(estimator=model_xgb, 
                        param_grid=xgb_params_test,
                        scoring='neg_root_mean_squared_error',
                        cv=5,
                        verbose=1,
                        n_jobs=-1)





def calc_model(x_train,y_train, x_valid, y_valid, x_test, fold, model_name):
    
    if model_name == 'catboost':
        model = cb.CatBoostRegressor(**cb_params)
        model.fit(x_train, y_train)
        y_valid_pred = model.predict(x_valid)
        y_pred = model.predict(x_test)
        model.save_model(f'{fold}')
        return model, y_valid_pred, y_pred
    elif model_name == 'lighgbm':
        train_set = lgb.Dataset(x_train, label=y_train)
        valid_set = lgb.Dataset(x_valid, label=y_valid)
        
        model = lgb.train(lgb_params, train_set, valid_sets=valid_set, 
                              categorical_feature=[], verbose_eval=500, early_stopping_rounds=200)
        y_valid_pred = model.predict(x_valid, num_iteration=model.best_iteration)
        y_pred = model.predict(x_test, num_iteration=model.best_iteration)
        model.save_model(f'{fold}')
        return model, y_valid_pred, y_pred
    elif model_name == 'xgboost':
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(x_train, y_train)
        y_valid_pred = model.predict(x_valid)
        y_pred = model.predict(x_test)
        model.save_model(f'{fold}')
        return model, y_valid_pred, y_pred
    else:
        print('错误的模型')
    return



model_ls = []

def KFold_func(model, x_train_, y_train_, x_test_, seed=1):
    K_folds = 5
    kf = KFold(n_splits=K_folds, shuffle=True, random_state=seed)
    y_pred = np.zeros(x_test_.shape[0])
    for fold, (train_index, valid_index) in (enumerate(kf.split(x_train_))):
        x_train, x_valid = x_train_.iloc[train_index], x_train_.iloc[valid_index]
        y_train, y_valid = y_train_.iloc[train_index], y_train_.iloc[valid_index]
        
        model = calc_model(x_train,y_train, x_valid, x_test_, '<model_name>')
        model.fit(x_train, y_train)
        y_valid_ = model.predict(x_valid, num_iteration=model.best_iteration)
        score = calc_acc(y_valid_, y_valid)
        print(score)
        y_pred_ = model.predict(x_test_, num_iteration=model.best_iteration)

        model.save_model(f'model_{fold}')
        model_ls.append(model)
        y_pred += y_pred_ / len(model_ls)

    return y_pred 


         

if __name__ == '__main__':
    model = Catboost()
    model.get_params()




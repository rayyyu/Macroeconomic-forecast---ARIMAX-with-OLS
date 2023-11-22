import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings('ignore')


factor_number_ols = 3
factor_number_olsAR = 5
factor_number_arimax = 7
# 滚动窗口长度
window_length = 4
ols_saves = pd.ExcelWriter(r'C:\Users\Ray S Yu\Desktop\ForecastStream\ARIMAX_sh_forecast_0824.xlsx')
rmse_save = pd.ExcelWriter(r'C:\Users\Ray S Yu\Desktop\ForecastStream\ARIMAX_sh_rmse_score.xlsx')


def buildLagLeadFeatures(s, lag, lead, dropna=True):
    # '''
    # Builds a new DataFrame to facilitate regressing over all possible lagged features
    # '''
    if type(s) is pd.DataFrame:
        new_dict = {}
        transfer_dict = {}
        for col_name in s:
            new_dict[col_name] = s[col_name]
            # create lagged Series
            if (lag > 0):
                for l in range(1, lag + 1):
                    new_dict['%s_lag%d' % (col_name, l)] = s[col_name].shift(l)
                    transfer_dict['%s_lag%d' % (col_name, l)] = col_name
            if (lead > 0):
                for j in range(1, lead + 1):
                    new_dict['%s_lead%d' % (col_name, j)] = s[col_name].shift(-j)
                    transfer_dict['%s_lead%d' % (col_name, j)] = col_name
        res = pd.DataFrame(new_dict, index=s.index)
    else:
        print('Only works for DataFrame')
        return None
    if dropna:
        return res.dropna(), transfer_dict
    else:
        return res, transfer_dict


# 对原始数据进行Transformation和Lag及Lead处理
def buildFeatures(macro_df, lag, lead, dropna=True):
    macro_temp = macro_df
    macro_temp1 = macro_temp
    ind_pd_raw, varsign_dict = buildLagLeadFeatures(macro_temp1, lag, lead, dropna)
    return ind_pd_raw, varsign_dict

def summarydata(df, features, columns_to_aggregate, aggregate_method):
    output = df.groupby(features)[columns_to_aggregate].agg(aggregate_method)
    return output


def vifTest(ind_Candidate_df):
    X = sm.add_constant(ind_Candidate_df)
    viftest_list = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return viftest_list


def multivariateLRTest(dep_t, ind_raw, validation, ytrainfinder, ytestfinder, dept_name):
    vif = vifTest(ind_raw)
    # 进行线性回归
    X = sm.add_constant(ind_raw)
    model = sm.OLS(dep_t, X)
    LR = model.fit()
    rec = validation.iloc[0:1, :]
    rec = sm.add_constant(rec, has_constant='add')
    rec_pred = LR.predict(rec)
    validation[f'{dept_name}_lag1'][1] = rec_pred
    validation[f'{dept_name}_lag2'][2] = rec_pred
    rec = validation.iloc[0:2, :]
    rec = sm.add_constant(rec, has_constant='add')
    rec_pred = LR.predict(rec)
    validation[f'{dept_name}_lag1'][2] = rec_pred[-1]
    validation = sm.add_constant(validation, has_constant='add')
    predict_value = pd.DataFrame(LR.predict(validation))
    predict_value.reset_index(drop=True, inplace=True)
    predict_value.iloc[0] = predict_value.iloc[0] + ytrainfinder.iloc[-1]
    predict_value = np.cumsum(predict_value)
    residual = np.sum((ytestfinder.to_numpy() - predict_value.to_numpy())**2)
    return residual


def ypredict(dep_t, ind_raw, validation):
    # 进行线性回归
    X = sm.add_constant(ind_raw)
    model = sm.OLS(dep_t, X)
    LR = model.fit()
    validation = sm.add_constant(validation, has_constant='add')
    predict_value = pd.DataFrame(LR.predict(validation))
    return predict_value


def ypredict_undiff(dep_t, ind_raw, validation, ytrainfinder):
    # 进行线性回归
    X = sm.add_constant(ind_raw)
    model = sm.OLS(dep_t, X)
    LR = model.fit()
    validation = sm.add_constant(validation, has_constant='add')
    predict_value = pd.DataFrame(LR.predict(validation))
    predict_value.reset_index(drop=True, inplace=True)
    predict_value.iloc[0] = predict_value.iloc[0] + ytrainfinder.iloc[-1]
    predict_value = np.cumsum(predict_value)
    return predict_value


def multivariateTest(model, dep_t, ind_raw, validation, ytest):
    if model == 'LR':
        return (multivariateLRTest(dep_t, ind_raw, validation, ytest))
    else:
        # 完善如Lasso等回归方式
        return ('Only LR supported for now')


dep_data = pd.read_excel(r'C:\Users\Ray S Yu\Desktop\macro datasets\上农因子库_上海.xlsx', sheet_name="dept_sh", index_col=0)
dep_data = dep_data.interpolate(method='linear', limit_direction='both', axis=0)
indept_data = pd.read_excel(r'C:\Users\Ray S Yu\Desktop\macro datasets\上农因子库_上海.xlsx', sheet_name="regressor", index_col=0)
indept_data = indept_data.interpolate(method='linear', limit_direction='both', axis=0)
dep_sh = pd.read_excel(r'C:\Users\Ray S Yu\Desktop\macro datasets\上农因子库_上海.xlsx', sheet_name="dept_sh", index_col=0)
dep_sh = dep_sh.interpolate(method='linear', limit_direction='both', axis=0)
xpredict = pd.read_excel(r'C:\Users\Ray S Yu\Desktop\macro datasets\xtest_arimax_上农.xlsx', index_col=0)

# 上海地区主要因子数据准备
date = '2014-03-31'
dep_data_all = dep_sh[dep_sh.index >= date]
dep_finder = dep_data_all.iloc[1:]
dep_data_diff = dep_sh.diff(1).dropna()
indept_data = indept_data.diff(1).dropna()
indept_data_sh = indept_data[indept_data.index >= date]
data = pd.concat([dep_data_diff, indept_data_sh], axis=1)
sh_lags, sh_lag_dict = buildFeatures(dep_data_all, lag=4, lead=0, dropna=True)
sh_list = [col for col in sh_lags]
backup_data, backup_dict = buildFeatures(indept_data_sh, lag=4, lead=0,
                                         dropna=True)
xpredict = xpredict.diff(1).dropna()
xpredict, test_dict = buildFeatures(xpredict, lag=4, lead=0,
                                    dropna=False)
xpredict = xpredict.iloc[-5:]
indept_list = [col for col in backup_data]
k = list(dep_data_all.columns)
sh_list = [i for i in sh_list if i not in k]
ind_pd_raw, varsign_dict = buildFeatures(data, lag=4, lead=0,
                                         dropna=True)
sh_list = [col for col in ind_pd_raw]
sh_list = [i for i in sh_list if i not in k]
list_y = dep_data_diff.columns
obs = np.sort(ind_pd_raw.index.unique())
cv = TimeSeriesSplit(n_splits=window_length, test_size=3, gap=0)

# 上海因子映射表
ind_dict = {'sh_v0': 'CPI', 'sh_v1': 'GDP_cum', 'sh_v2': 'IND', 'sh_v3': 'CPI',
            'sh_v4': 'IND', 'sh_v5': 'FIN', 'sh_v6': 'RET', 'sh_v7': 'RET', 'sh_v8': 'PPI',
            'sh_v9': 'PPI', 'sh_v10': 'PPI', 'sh_v11': 'PPI', 'sh_v12': 'EX', 'sh_v13': 'EX',
            'sh_v14': 'GDP_cum', 'sh_v15': 'GDP_cum', 'sh_v16': 'FIX', 'sh_v17': 'EX', 'sh_v18': 'EX',
            'sh_v19': 'GDP', 'sh_v20': 'GDP_cum', 'sh_v21': 'IND', 'sh_v22': 'IND', 'sh_v23': 'RET',
            'sh_v24': 'RET', 'sh_v25': 'FIN', 'sh_v26': 'FIN', 'sh_v27': 'RET', 'sh_v28': 'RET'}
model_dict = {}
final_save = pd.DataFrame()

models = pd.read_excel(r'C:\Users\Ray S Yu\Desktop\ARIMAX_sh_bestmodel_0823.xlsx', index_col=0)
model_list = list(models['VarList'])[0::8]
model_dict = dict(zip(k, model_list))
final_forecast = pd.DataFrame()
for i in range(len(list_y)):
    forecast_dept = [list_y[i]][0].replace("'", "")
    forecast_indept = model_dict.get(forecast_dept)
    forecast_indept = forecast_indept.replace("[", "")
    forecast_indept = forecast_indept.replace("]", "")
    forecast_indept = forecast_indept.replace("'", "")
    forecast_indept = list(forecast_indept.split(", "))
    validation = xpredict[forecast_indept]
    full_x = ind_pd_raw[forecast_indept]
    ytrain = pd.DataFrame(dep_data_diff[[list_y[i]]])
    ytrain = ytrain.iloc[4:]
    forecast = ypredict(ytrain, full_x, validation.iloc[0:1, :])
    xpredict[f'{list_y[i]}_lag1'][1] = forecast.iloc[0, 0]
    xpredict[f'{list_y[i]}_lag2'][2] = forecast.iloc[0, 0]
    xpredict[f'{list_y[i]}_lag3'][3] = forecast.iloc[0, 0]
    xpredict[f'{list_y[i]}_lag4'][4] = forecast.iloc[0, 0]

for i in range(len(list_y)):
    forecast_dept = [list_y[i]][0].replace("'", "")
    forecast_indept = model_dict.get(forecast_dept)
    forecast_indept = forecast_indept.replace("[", "")
    forecast_indept = forecast_indept.replace("]", "")
    forecast_indept = forecast_indept.replace("'", "")
    forecast_indept = list(forecast_indept.split(", "))
    validation = xpredict[forecast_indept]
    full_x = ind_pd_raw[forecast_indept]
    ytrain = pd.DataFrame(dep_data_diff[[list_y[i]]])
    ytrain = ytrain.iloc[4:]
    forecast = ypredict(ytrain, full_x, validation.iloc[0:2, :])
    xpredict[f'{list_y[i]}_lag1'][2] = forecast.iloc[1, 0]
    xpredict[f'{list_y[i]}_lag2'][3] = forecast.iloc[1, 0]
    xpredict[f'{list_y[i]}_lag3'][4] = forecast.iloc[1, 0]

for i in range(len(list_y)):
    forecast_dept = [list_y[i]][0].replace("'", "")
    forecast_indept = model_dict.get(forecast_dept)
    forecast_indept = forecast_indept.replace("[", "")
    forecast_indept = forecast_indept.replace("]", "")
    forecast_indept = forecast_indept.replace("'", "")
    forecast_indept = list(forecast_indept.split(", "))
    validation = xpredict[forecast_indept]
    full_x = ind_pd_raw[forecast_indept]
    ytrain = pd.DataFrame(dep_data_diff[[list_y[i]]])
    ytrain = ytrain.iloc[4:]
    forecast = ypredict(ytrain, full_x, validation.iloc[0:3, :])
    xpredict[f'{list_y[i]}_lag1'][3] = forecast.iloc[2, 0]
    xpredict[f'{list_y[i]}_lag2'][4] = forecast.iloc[2, 0]

for i in range(len(list_y)):
    forecast_dept = [list_y[i]][0].replace("'", "")
    forecast_indept = model_dict.get(forecast_dept)
    forecast_indept = forecast_indept.replace("[", "")
    forecast_indept = forecast_indept.replace("]", "")
    forecast_indept = forecast_indept.replace("'", "")
    forecast_indept = list(forecast_indept.split(", "))
    validation = xpredict[forecast_indept]
    full_x = ind_pd_raw[forecast_indept]
    ytrain = pd.DataFrame(dep_data_diff[[list_y[i]]])
    ytrain = ytrain.iloc[4:]
    forecast = ypredict(ytrain, full_x, validation.iloc[0:4, :])
    xpredict[f'{list_y[i]}_lag1'][4] = forecast.iloc[3, 0]



for i in range(len(list_y)):
    forecast_dept = [list_y[i]][0].replace("'", "")
    forecast_indept = model_dict.get(forecast_dept)
    forecast_indept = forecast_indept.replace("[", "")
    forecast_indept = forecast_indept.replace("]", "")
    forecast_indept = forecast_indept.replace("'", "")
    forecast_indept = list(forecast_indept.split(", "))
    validation = xpredict[forecast_indept]
    full_x = ind_pd_raw[forecast_indept]
    ytrain = pd.DataFrame(dep_data_diff[[list_y[i]]])
    ytrain = ytrain.iloc[4:]
    ytrainfinder = pd.DataFrame(dep_finder[[list_y[i]]])
    ytrainfinder = np.transpose(ytrainfinder)
    ytrainfinder.reset_index(drop=True, inplace=True)
    ytrainfinder = np.transpose(ytrainfinder)
    forecast = ypredict_undiff(ytrain, full_x, validation, ytrainfinder)
    final_forecast = pd.concat([final_forecast, forecast], axis=1)

final_forecast.columns = dep_data_all.columns
final_forecast.to_excel(ols_saves)
ols_saves.save()
dicts = pd.DataFrame(model_dict.items())
dict_save = pd.ExcelWriter(r'C:\Users\Ray S Yu\Desktop\ForecastStream\ARIMAX_sh_dict.xlsx')
dicts.to_excel(dict_save)
dict_save.save()
print(obs)
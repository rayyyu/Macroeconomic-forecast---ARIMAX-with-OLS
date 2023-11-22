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
# Length of rolling window
window_length = 4
ols_saves = pd.ExcelWriter(r'C:\Users\Ray S Yu\Desktop\ForecastStream\ARIMAX_sh_0823.xlsx')
rmse_save = pd.ExcelWriter(r'C:\Users\Ray S Yu\Desktop\ForecastStream\ARIMAX_sh_rmse_0823.xlsx')


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


# Lag/Lead transform the input data

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
    X = sm.add_constant(ind_raw)
    model = sm.OLS(dep_t, X)
    LR = model.fit()
    validation = sm.add_constant(validation, has_constant='add')
    predict_value = pd.DataFrame(LR.predict(validation))
    return predict_value


def ypredict_undiff(dep_t, ind_raw, validation, ytrainfinder):
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
        return ('Only LR supported for now')


dep_data = pd.read_excel(r'C:\Users\Ray S Yu\Desktop\macro datasets\上农因子库_上海.xlsx', sheet_name="dept_sh", index_col=0)
dep_data = dep_data.interpolate(method='linear', limit_direction='both', axis=0)
indept_data = pd.read_excel(r'C:\Users\Ray S Yu\Desktop\macro datasets\上农因子库_上海.xlsx', sheet_name="regressor", index_col=0)
indept_data = indept_data.interpolate(method='linear', limit_direction='both', axis=0)
dep_sh = pd.read_excel(r'C:\Users\Ray S Yu\Desktop\macro datasets\上农因子库_上海.xlsx', sheet_name="dept_sh", index_col=0)
dep_sh = dep_sh.interpolate(method='linear', limit_direction='both', axis=0)
xpredict = pd.read_excel(r'C:\Users\Ray S Yu\Desktop\macro datasets\xtest_arimax_上农.xlsx', index_col=0)

# Data Cleaning

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
xpredict = xpredict.iloc[-3:]
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

ind_dict = {'sh_v0': 'CPI', 'sh_v1': 'GDP_cum', 'sh_v2': 'IND', 'sh_v3': 'CPI',
            'sh_v4': 'IND', 'sh_v5': 'FIN', 'sh_v6': 'RET', 'sh_v7': 'RET', 'sh_v8': 'PPI',
            'sh_v9': 'PPI', 'sh_v10': 'PPI', 'sh_v11': 'PPI', 'sh_v12': 'EX', 'sh_v13': 'EX',
            'sh_v14': 'GDP_cum', 'sh_v15': 'GDP_cum', 'sh_v16': 'FIX', 'sh_v17': 'EX', 'sh_v18': 'EX',
            'sh_v19': 'GDP', 'sh_v20': 'GDP_cum', 'sh_v21': 'IND', 'sh_v22': 'IND', 'sh_v23': 'RET',
            'sh_v24': 'RET', 'sh_v25': 'FIN', 'sh_v26': 'FIN', 'sh_v27': 'RET', 'sh_v28': 'RET', 'sh_v29': 'FIN',
            'sh_v30': 'GDP'}
model_dict = {}
final_save = pd.DataFrame()
for i in tqdm(range(len(list_y)), ascii=True):
    sheet_name = [list_y[i]][0].replace("'", "")
    sh_list_tmp = sh_list.copy()
    full_x = ind_pd_raw[sh_list_tmp]
    cast_ind = ind_dict.get(sheet_name)
    ind_pd = ind_pd_raw[sh_list_tmp]
    a = full_x
    a = a.drop([f'{sheet_name}_lag1', f'{sheet_name}_lag2', f'{sheet_name}_lag3', f'{sheet_name}_lag4'], axis=1)
    possible_comb = []
    for s in combinations(a.columns, 3):
        if f'{cast_ind}' in s:
            s = s + tuple([f'{sheet_name}_lag1', f'{sheet_name}_lag2', f'{sheet_name}_lag3', f'{sheet_name}_lag4'])
            possible_comb.append(s)
    final_result = []
    for train_index, test_index in cv.split(obs):
        train_obs, test_obs = obs[train_index], obs[test_index]
        ytrain, ytest = pd.DataFrame(dep_data_diff[[list_y[i]]].loc[train_obs]), pd.DataFrame(dep_data_diff[[list_y[i]]].loc[test_obs])
        ytrainfinder, ytestfinder = pd.DataFrame(dep_finder[[list_y[i]]].loc[train_obs]), pd.DataFrame(
            dep_finder[[list_y[i]]].loc[test_obs])
        ytest.columns = range(ytest.shape[1])
        ytest.reset_index(drop=True, inplace=True)
        xtrain, xtest = pd.DataFrame(ind_pd.loc[train_obs]), pd.DataFrame(ind_pd.loc[test_obs])
        ind_name_total = []
        coef_total = []
        resid_total = []
        model_total = []
        var_total = []
        model_no = 1
        for j in tqdm(range(len(possible_comb)), ascii=True):
            var_tmp = list(possible_comb[j])
            var_total.extend([list(possible_comb[j])] * (factor_number_arimax + 1))
            ind_temp = xtrain[var_tmp]
            x_temp = xtest[var_tmp]
            ind_name_temp = ['Intercept'] + var_tmp
            ytrainfinder = np.transpose(ytrainfinder)
            ytrainfinder.reset_index(drop=True, inplace=True)
            ytrainfinder = np.transpose(ytrainfinder)
            dep = list_y[i]
            resid_tmp = multivariateLRTest(ytrain, ind_temp, x_temp, ytrainfinder, ytestfinder, dep)
            ind_name_total.extend(ind_name_temp)
            resid_total.extend([resid_tmp] * (factor_number_arimax + 1))
            model_total.extend([model_no] * (factor_number_arimax + 1))
            model_no = model_no + 1
        result_df = pd.DataFrame({'Model_no': model_total,
                              'Variable': ind_name_total,
                              'Resid': resid_total,
                              'VarList': var_total})
        final_result.append(result_df)
    final_result = pd.concat(final_result, axis=0)
    comp = summarydata(final_result, ['Model_no'], ['Resid'], 'sum')
    best_model = comp[comp.Resid == comp.Resid.min()]
    best_model['model'] = best_model.index
    final_model = final_result.loc[final_result['Model_no'] == best_model.iloc[0, 1]]
    final_model['RMSE'] = np.sqrt(final_model.loc[:, 'Resid'].mean())
    final_model = final_model.iloc[-8:]
    final_save = pd.concat([final_save, final_model], ignore_index=True)
    final_save.to_excel(rmse_save)
    rmse_save.save()
    best_var = final_model['VarList'].iloc[0]
    model_dict[sheet_name] = best_var

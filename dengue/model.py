import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LinearRegression
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.genmod.cov_struct import Independence
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import math



def calculate_stats(train_df):
    X = add_constant(train_df)
    #print (X.head(10))
    #print (X.dtypes)
    vif_cols = pd.Series([variance_inflation_factor(X.values,i) for i in range(X.shape[1])],index=X.columns)

    print (vif_cols)



def pre_processing(train_df,test_df):

    print ('In pre-processing')

    # Drop for vif
    train_df.drop('city', inplace=True, axis=1)
    train_df.drop('week_start_date', inplace=True, axis=1)

    # cols with high vif > 10
    train_df.drop('precipitation_amt_mm', inplace=True, axis=1)
    train_df.drop('reanalysis_air_temp_k', inplace=True, axis=1)
    train_df.drop('reanalysis_specific_humidity_g_per_kg', inplace=True, axis=1)
    train_df.drop('reanalysis_dew_point_temp_k', inplace=True, axis=1)
    train_df.drop('reanalysis_avg_temp_k', inplace=True, axis=1)
    train_df.drop('station_avg_temp_c', inplace=True, axis=1)
    train_df.drop('weekofyear', inplace=True, axis=1)
    train_df.drop('reanalysis_max_air_temp_k', inplace=True, axis=1)

    # Drop for vif
    test_df.drop('city', inplace=True, axis=1)
    test_df.drop('week_start_date', inplace=True, axis=1)

    # cols with high vif > 10
    test_df.drop('precipitation_amt_mm', inplace=True, axis=1)
    test_df.drop('reanalysis_air_temp_k', inplace=True, axis=1)
    test_df.drop('reanalysis_specific_humidity_g_per_kg', inplace=True, axis=1)
    test_df.drop('reanalysis_dew_point_temp_k', inplace=True, axis=1)
    test_df.drop('reanalysis_avg_temp_k', inplace=True, axis=1)
    test_df.drop('station_avg_temp_c', inplace=True, axis=1)
    test_df.drop('weekofyear', inplace=True, axis=1)
    test_df.drop('reanalysis_max_air_temp_k', inplace=True, axis=1)

    # Impute missing columns - mean
    imp_mean = SimpleImputer(missing_values=np.nan,strategy='mean')
    imp_mean.fit(train_df)
    train_df_np = imp_mean.transform(train_df)

    imp_mean.fit(test_df)
    test_df_np = imp_mean.transform(test_df)

    train_df_2 = pd.DataFrame(train_df_np,columns = train_df.columns)
    test_df_2 = pd.DataFrame(test_df_np,columns=test_df.columns)

    return train_df_2, test_df_2


def poisson_reg(train_df,test_df):

    y = train_df.total_cases
    train_df.drop('total_cases',axis=1,inplace=True)
    train_df = add_constant(train_df)

    print (y.head(10))
    print (train_df.head(10))

    poisson_model = sm.Poisson(y,train_df).fit()
    preds = poisson_model.predict(train_df)
    diff = abs(preds - y)
    print (preds.head(10))
    print (diff.head(10))
    print (np.mean(diff))




def eda(_df):

    print (_df.ndvi_ne.head(10))
    print(_df.ndvi_ne.describe())


def main():

    train_file = '/home/nitin/Desktop/dengue/dengue_features_train.csv'
    test_file = '/home/nitin/Desktop/dengue/dengue_features_test.csv'
    train_labels_file = '/home/nitin/Desktop/dengue/dengue_labels_train.csv'

    train_df = pd.read_csv(train_file)
    train_labels_df = pd.read_csv(train_labels_file)
    train_data_merged = train_labels_df.merge(train_df,left_on=['city','year','weekofyear'],right_on=['city','year','weekofyear'],how='left')
    test_df = pd.read_csv(test_file)

    sj_data = train_data_merged[train_data_merged.city == 'sj']
    ig_data = train_data_merged[train_data_merged.city == 'ig']

    sj_data_test = test_df[test_df.city == 'sj']
    ig_data_test = test_df[test_df.city == 'ig']

    sj_data['month'] = sj_data['week_start_date'].str.split("-").apply(lambda z: z[1]).map(int)
    ig_data['month'] = ig_data['week_start_date'].str.split("-").apply(lambda z: z[1]).map(int)
    sj_data_test['month'] = sj_data_test['week_start_date'].str.split("-").apply(lambda z: z[1]).map(int)
    ig_data_test['month'] = ig_data_test['week_start_date'].str.split("-").apply(lambda z: z[1]).map(int)

    train_data, test_data = pre_processing(sj_data,sj_data_test)
    poisson_reg(train_data,test_data)




    #calculate_stats(sj_data_2)

    #sj_data_2.add
    #poisson_reg(sj_data_2,None)









main()


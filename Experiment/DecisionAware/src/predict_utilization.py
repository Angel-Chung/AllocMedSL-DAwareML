
import sys
from sklearn.ensemble import RandomForestRegressor
from src.preprocess import fetch_preprocessedDHIS2_EssMeds
from src.run_ML import run_ML_diff_test_dates
from src.create_features import create_features_essential_meds
from retina.metrics import *
#from feature_engineering.features import *
#from DHIS2.loading import *
from conf.config import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def predict_utilization(dates: list = ['2100-01-01'], lead_time=2,
                        date_column='date', target_col='target',
                        output_columns=['hf_pk', 'pred', 'product',
                                        'target','date','avg_3months_DHIS2'],
                        aggregation=None):
    """run forecasting model and predict utilization"""
    print('dates',dates)
    # fetch preprocessed data
    df = fetch_preprocessedDHIS2_EssMeds(
        preprocessed_file_name='pivot_from_raw_DHIS2')

    if aggregation == None:
        # create features
        df4ML, mapping = create_features_essential_meds(df, lead_time=lead_time, return_mapping=True)
        cat_feat = ['product']

    
    elif aggregation=='district':
        feat_agg = ['date','district', 'product', 'avg_3months_DHIS2',
            'quantity','num_fac_per_district']
        df = df.merge(df.groupby('district')[
            'fac_name'].nunique().reset_index().rename(columns={
                'fac_name':'num_fac_per_district'}), on='district')[feat_agg]
        df = df.groupby(['date', 'district', 
                                 'product','num_fac_per_district'])[[
                                     'quantity', 'avg_3months_DHIS2']].mean().astype('int').reset_index()
        output_columns = [i for i in output_columns if i!='hf_pk'] + ['district']
        # create features
        df4ML, mapping = create_features_essential_meds(df, grouping_cols=[aggregation, 'product'] ,
                                                        lead_time=lead_time, return_mapping=True)
        cat_feat = ['product', 'district']
    else:
        raise ValueError(f'{aggregation} method is not implemented!')
 
    # modeling
    hyperparams = {
        'random_state': 10,
        'max_depth': None,
        'n_estimators': 500,
        'max_features': 'auto',
        'min_samples_leaf': 8,
        'n_jobs': -1
    }
    model_name = 'RF'
    d = {model_name: RandomForestRegressor(**hyperparams)}
    results, train, val = run_ML_diff_test_dates(df4ML, d, dates,
                                     date_column, target_col, lead_time)
    train.to_csv(f'data/train.csv')
    val.to_csv(f'data/val.csv')
    final_res = results[model_name]
    # replace coded features to original categories
    for i in cat_feat:
        final_res[i] = final_res[i].replace(mapping[i])
    
    return final_res[output_columns]

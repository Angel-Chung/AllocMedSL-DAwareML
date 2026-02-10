import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV, Ridge, Lasso, LassoCV
import warnings
warnings.filterwarnings('ignore')

def get_training_data(df: pd.DataFrame):
    """
    Args:
        df: Train dataset 
    """

    drop_cols = ['date', 'target']
    if 'weight' in df.columns:
        drop_cols.append('weight')
    X = df.drop(drop_cols, axis = 1)
    y = df['target']
    if 'weight' in df.columns:
        w = df['weight']
    else:
        w = None
    return X, y, w

def train_model(df: pd.DataFrame, n_estimators):
    """
    Args:
        df: Train dataset 
    """

    X_train, y_train, w_train = get_training_data(df)
    
    if 'weight' in df.columns:
        # synthetic 8
        #hyperparams = {
            #'random_state': 10,
            #'max_depth': 41,
            #'min_samples_split':6,
            #'n_estimators': n_estimators,
            #'max_features': 'auto',
            #'min_samples_leaf': 100,
            #'n_jobs': -1,
            #'bootstrap': False
            #'criterion':'absolute_error'
        #}
        # 8 products original weight
        hyperparams = {
            'random_state': 10,
            'max_depth': 17,
            'min_samples_split': 2,
            'n_estimators': n_estimators,
            'max_features': 'log2',#log2
            'min_samples_leaf': 100,
            'n_jobs': -1,
            'bootstrap': False
            #'criterion':'absolute_error'
        }
    else:
        hyperparams = {
            'random_state': 10,
            'max_depth': 17,
            'n_estimators': n_estimators,
            'max_features': 'sqrt',
            'min_samples_leaf': 100,
            'n_jobs': -1,
            'bootstrap': False
            #'criterion':'absolute_error'
        }
    
    rfr = RandomForestRegressor(**hyperparams)
    rfr.fit(X_train, y_train, w_train)

    return rfr

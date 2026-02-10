"""this is a wrapper to run ML"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RandomizedSearchCV
import warnings


def run_ML_diff_test_dates(df: pd.DataFrame, model: dict, test_dates: list,
                           date_column: str, target_col: str, lead_time=1,
                           nonlinear_agg=False, learn_hyperparams=False):
    """train ML model and test on the provided test dates

    Parameters
    ----------
    df : d.DataFrame
        dataframe including train and test sets
    model: dict
        dictionary of the sklearn model with hyperparams  
        with a predefined name, e.g {'rf_reg':RandomForestRegressor(*params)}
    test_dates: list
        dates to calculate model performance (test set)
    date_column: str
        name of date column
    lead_time: int
        prediction horizon, 1 is next month, 2 is next 2 month, etc.
    nonlinear_agg: boolean
        learn nonlinear aggregation over different trees from random forest
    learn_hyperparams: boolean
        run hyperoptimization before fitting the model
    Returns:
    ---------
    dictionary of the predicted and actual values in the test sets
    """
    df = df.copy()
    results = {}
    for i in range(len(test_dates)):
        date = test_dates[i]
        print('date',date)
        train = df[(df[date_column] + pd.offsets.DateOffset(
            months=(lead_time-1))) < date]
        if train.isna().sum().sum() > 0:
            train = train.dropna()
            warnings.warn('data include Nan values. Dropped from train sets!')
        print(f"Training min: {train['date'].min()}")
        print(f"Training max: {train['date'].max()}")
        val = df[df['date'] == date]
        print(f"Test min: {val['date'].min()}")
        print(f"Test max: {val['date'].max()}")
        print(f"Total sample test: {len(val)}")
        # , 'district_cat', 'fac_name_cat']
        drop_cols = [date_column, target_col]
        X_train = train.drop(drop_cols, axis=1)
        y_train = train[target_col]
        X_val = val.drop(drop_cols, axis=1)
        y_val = val[target_col]
        for idx, (k, v) in enumerate(model.items()):
            if learn_hyperparams and k == 'RF':
                params = find_hperparams_rf(X_train, y_train)
                model_tmp = RandomForestRegressor(n_jobs=-1, **params)
            else:
                model_tmp = v
            model_tmp.fit(X_train, y_train)
            preds = np.int64(model_tmp.predict(X_val))
            if nonlinear_agg:
                nonlinear_agg_pred = aggregate_learned_trees(model_tmp, train,
                                                             val, test_date=date,
                                                             target_col=target_col,
                                                             date_col=date_column,
                                                             model='rf', num_month_valid=3)
            if i == 0:
                res = val.copy()
                res['pred'] = preds
                if nonlinear_agg:
                    res['nonlinear_agg_pred'] = nonlinear_agg_pred
                results[k] = res
            else:
                tmp = val.copy()
                tmp['pred'] = preds
                if nonlinear_agg:
                    tmp['nonlinear_agg_pred'] = nonlinear_agg_pred
                results[k] = pd.concat(
                    [results[k], tmp]).reset_index(drop=True)
            results[k+'_lastmodel'] = model_tmp
    return results, train, val


def aggregate_learned_trees(trained_model, df_train, df_test, test_date,
                            target_col, date_col, model='rf', num_month_valid=6):
    """train extra model to map trees output to the target"""
    val_date = [pd.to_datetime(test_date) - pd.DateOffset(months=i+1)
                for i in range(num_month_valid)]
    X_test = df_test.drop([target_col, date_col], axis=1)
    df_valid = df_train[df_train.date.isin(val_date)]
    X_valid = df_valid.drop([target_col, date_col], axis=1)
    pred_per_tree = np.stack([t.predict(X_valid.to_numpy())
                             for t in trained_model.estimators_])
    if model == 'rf':
        m_reg = RandomForestRegressor(n_jobs=-1, n_estimators=300)
    elif model == 'LR':
        m_reg = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1], normalize=False)
    elif model == 'lgb':
        m_reg = lgb.LGBMRegressor()

    m_reg.fit(pred_per_tree.T, df_valid[target_col].values)
    # test
    pred_per_tree_test = np.stack(
        [t.predict(X_test.to_numpy()) for t in trained_model.estimators_])
    pred_multi_learning = m_reg.predict(pred_per_tree_test.T)
    return pred_multi_learning


def find_hperparams_rf(X_train, y_train):
    """find best hyperparams for the model"""
    n_estimators = [50, 100, 500]  # number of trees in the random forest
    # number of features in consideration at every split
    max_features = ['auto', 'sqrt']
    # maximum number of levels allowed in each decision tree
    max_depth = [int(x) for x in np.linspace(10, 120, num=12)]
    min_samples_split = [2, 6, 10, 20]  # minimum sample number to split a node
    # minimum sample number that can be stored in a leaf node
    min_samples_leaf = [1, 3, 4, 10]
    bootstrap = [True, False]  # method used to sample data points

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestRegressor()

    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   n_iter=100, cv=5, verbose=2, random_state=35, n_jobs=-1)
    rf_random.fit(X_train, y_train)
    return rf_random.best_params_

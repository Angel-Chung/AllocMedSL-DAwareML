import numpy as np
import pandas as pd

from dateutil.relativedelta import relativedelta

from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def APE(y1, y2, ignore_zeros=True):
    """
    Absolute Percentage Error
    """
    y1, y2 = np.array(y1), np.array(y2)

    if ignore_zeros:
        mask = y2 != 0
        y2 = y2[mask]
        y1 = y1[mask]
        y_true = y2
    else:
        y_true = y2.copy()
        mask = y2 == 0
        y_true[mask] = 1.

    return np.array(np.abs(y1-y2) / np.abs(y_true))


def WMAPE(y_true: pd.Series, y_pred: np.ndarray, grouping_col=None):
    """
    Weighted mean absolute percentage error
    WMAPE = sum(abs(y_true - y_pred)/sum(y_true))


    Args:
        y_true: pd.Series of true 'y' values
        y_pred: np.array of output from predictive models

    Returns:
       float number of WMAPE in percentage
    """

    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)


def fetch_ts_metrics(y_true: pd.Series, y_pred: np.ndarray):
    """
    Fetches all relevant time series metrics for predictions

    Args:
        y_true: pd.Series of true 'y' values
        y_pred: np.array of output from predictive models

    Returns:
        Dictionary of results 
    """

    res = {

        'mean_absolute_error': mean_absolute_error(y_true, y_pred),
        'median_absolute_error': median_absolute_error(y_true, y_pred),
        'mean_squared_error': mean_squared_error(y_true, y_pred),
        'rmse': mean_squared_error(y_true, y_pred, squared=False),
        'r2': r2_score(y_true, y_pred),
        'median_abs_per_error': np.median(APE(y_pred, y_true)),
        'mean_abs_per_error': np.mean(APE(y_pred, y_true)),
        'weighted_mean_absolute_percentage_error': WMAPE(y_true, y_pred)
    }

    return res

"""
Store functions useful for modeling, such as metrics fetching/etc
"""

import numpy as np
import pandas as pd

from dateutil.relativedelta import relativedelta

def split_data(df: pd.DataFrame, date_column: str, frequency: str,
               val_periods: int, test_periods: int):
    """
    Function to split data into train/val/test.
    
    Args:
        df: dataframe to split
        date_column: name of date column
        frequency: Frequency to use to split data
        val_periods: # of periods to use as validation data
        test_periods: # of periods to use as a test set
    
    Returns:
        (train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame)
    """
    
    # ensuring date column is in datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    if frequency == 'M': 
        cutoff_train = df[date_column].max() - relativedelta(months = test_periods + val_periods - 1)
        cutoff_val = df[date_column].max() - relativedelta(months = test_periods)
    
    elif frequency == '2W':
        cutoff_train = df[date_column].max() - relativedelta(days = 14*(test_periods + val_periods) - 1)
        cutoff_val = df[date_column].max() - relativedelta(days = 14*test_periods) 
    
    elif frequency == 'W':
        cutoff_train = df[date_column].max() - relativedelta(days = 7*(test_periods + val_periods) - 1)
        cutoff_val = df[date_column].max() - relativedelta(days = 7*test_periods)
    
    elif frequency == '2M': 
        cutoff_train = df[date_column].max() - relativedelta(months = 2*(test_periods + val_periods) - 1)
        cutoff_val = df[date_column].max() - relativedelta(months = 2*(test_periods))

    train = df[df[date_column] < cutoff_train]
    test = df[df[date_column] > cutoff_val]
    val = df[(df[date_column] >= cutoff_train) & (df[date_column] <= cutoff_val)]

    return train, val, test
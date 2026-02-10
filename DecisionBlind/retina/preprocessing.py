
"""
Generic functions to preprocess
raw TS data into Ml-ready format
"""

import pandas as pd
import numpy as np

import datetime


def filter_date(max_date):
    """
    Check if date is end of period. If not return the closest end of period
    """

    date = None
    if 'M' in FREQUENCY:
        if not end_of_month(max_date):
            date = pd.to_datetime(str(max_date.year) + '-' + str(max_date.month) + '-01')
    
    elif 'W' in FREQUENCY:
        if not end_of_week(max_date):
            date = max_date - datetime.timedelta(days=max_date.weekday())
    
    return date


def end_of_month(date, num_days=5):
    """
    Function to check whether or not the date given to use 
    was at the end of the month (if we have data <num_days> before the end of the day
    safe to assume the data is complete)
    (https://stackoverflow.com/questions/31753384/check-if-it-is-the-end-of-the-month-in-python)
    
    """
    todays_month = date.month
    tomorrows_month = (date + datetime.timedelta(days=num_days)).month
    
    return True if tomorrows_month != todays_month else False

def end_of_week(date):
    """
    Function to check whether or not the date given to use 
    was close to the end of the week (if we have data till Friday safe to assume 
    the week is complete)
    """

    week_day = date.weekday()
    
    return week_day >= 4


def aggregate_ts(df: pd.DataFrame, date_col: str, grouping_cols: list, freq = str, return_count = False):
    """
    Generic function to aggregate time series data 
    
    Args:
        df: Pandas dataframe 
        date_col: Date column to use to aggregate
        grouping_cols: list of columns to group by (i.e. facility/product IDs)
        #~~quantity_col: column to aggregate over~~
        freq: string of period (e.g. D= daily, W = weekly, etc
        return_count: boolean for whether ro return the count. Default: return sum
    
    Returns: aggregated Pandas dataframe
    """
    
    df = df.copy()
    
    # setting index to datetime column
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(df[date_col])
        
    # grouping and aggregating over target
    if return_count: 
        df = pd.DataFrame(df.groupby(grouping_cols + [pd.Grouper(key = date_col,freq = freq)])[target].count())
    else: 
        df = pd.DataFrame(df.groupby(grouping_cols + [pd.Grouper(key = date_col,freq = freq)])[target].sum())
  
    return df.reset_index() 



def reindex_ts(df: pd.DataFrame, freq: str, date_col: str, grouping_cols: list, max_date = False):
    """
    Function to add missing rows for each 'group', for example months 
    without any product sells
    
    Args:
        df: dataframe where each row is one period per group
        freq: frequency to reindex data (typically frequency of the df)
        date_col: name of the date column to form basis of reindexing
        grouping_cols: columns that are unique to each series (eg ID, name, product)
        max_date: boolean to use the max date of the DF (all series) or a single series
    
    Returns: re-indexed df
    """
    
    # setting date column as the index, after ensuring the column 
    # is in DT format
    df = df.copy()
    df[date_col] = df[date_col]
    max_date_dt = df[date_col].max()
    df = df.set_index(date_col)
    
    # applying re-indexing to each series
    print("using... ")
    if max_date:
        print("max_date")
        df = df.groupby(grouping_cols).apply(lambda x: reindex_alt(x, freq = freq, max_date = max_date_dt))
    else:
        print("not max date")
        df = df.groupby(grouping_cols).apply(lambda x: reindex_(x, freq = freq))
    
    # 'resetting' column names and index
    df = df.drop(grouping_cols, axis = 1)
    df = df.reset_index()
    df = df.drop('level_' + str(len(grouping_cols)), axis = 1)
    df = df.rename({'index': date_col}, axis = 1)
    
    return df

def reindex_(df: pd.DataFrame, freq: str):
    """
    Function that will automatically add in any periods completely missing a time series.
    Only adds in missing periods within minimum/maximum dates within the series.
    """
    
    dates = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    
    return df.reindex(dates).reset_index()

def reindex_alt(df: pd.DataFrame, freq: str, max_date):
    """
    Function that will automatically add in any periods completely missing a time series.
    Only adds in missing periods within minimum/maximum dates within the series.
    """
    
    dates = pd.date_range(df.index.min(), max_date, freq=freq)
    
    return df.reindex(dates).reset_index()

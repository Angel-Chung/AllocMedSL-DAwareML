"""
Generic functions to generate TS features from any time-series dataset,
including forecasting multiple time series simultaneously.

Data is assumed to have one entry for each period per group.

"""

import pandas as pd
import numpy as np

from pandas.api.types import is_string_dtype, is_numeric_dtype
from dateutil.relativedelta import relativedelta


def split_dates(df: pd.DataFrame, date_column: str):
    """
    Splits dates into numeric features:
        - year
        - month
        - dayofweek
        - weekofyear

    Args:
        df: pandas dataframe
        date_column: name of date column in dataframe

    Returns:
        df: Pandas dataframe with columns w/numeric features added
    """

    date_df = pd.DataFrame({"year": df[date_column].dt.year,
                            "month": df[date_column].dt.month,
                            "dayofweek": df[date_column].dt.dayofweek,
                            "weekofyear": df[date_column].dt.isocalendar().week,
                            })

    # add new date features to dataframe
    df = pd.concat([df, date_df], axis=1)

    return df


def str_to_cat(df, return_mapping=False):
    """
    Converts string/object features to categories, and replaces string with
    category code.

    Args:
        df: pandas dataframe
        return_mapping: boolean return mapping of categories codes and the original categories
    Returns:
        df: pandas dataframe
    """

    df = df.copy()
    mapping = {}
    for p, q in df.items():
        if is_string_dtype(q):
            df[p] = q.astype('category').cat.as_ordered()
            codes = df[p].cat.codes
            cats = df[p]
            mapping[p] = dict(zip(codes,cats))
            df[p] = codes
    if return_mapping:
        return df, mapping
    return df


def add_prev_periods(df: pd.DataFrame,
                     date_column: str,
                     grouping_columns: list,
                     value_column: str,
                     window: int,
                     output_name=None,
                     imputation_strat='mean'):

    """
    Add features for prior period(s) values

    Args:
        df: Pandas dataframe
        date_column: Date column to use for rolling window
        grouping_columns: columns to group by, usually product/facility
        value_column: column to use to create lagged values
        window: number of previous periods to add
        imputation_strat: default mean. You can change to 0/np.nan by adding these
        to this argument instead.

    Returns:
        df: df with previous period columns added
    """

    df = df.copy()

    orig = df.columns
    if output_name == None:
        output_name = value_column

    if imputation_strat == 'mean':
        # if mean is not in the columns, calculate it!
        if not output_name + '_mean' in orig:
            df = add_rolling(df, grouping_columns, date_column, value_column)

    # pre-sorting dataframe by date
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column)

    for i in range(0, window):
        # naming previous periods t-1, t-2, etc.
        i = i+1
        name = 't-' + str(i)
        df[name] = df.groupby(grouping_columns)[value_column].shift(i)

        # if the series length < window, then impute value using imputation_strat
        if imputation_strat == 'mean':
            df[name] = np.where(df[name].isnull(), df[value_column + '_mean'], df[name])
        else:
            df[name] = np.where(df[name].isnull(), imputation_strat, df[name])

    # if mean wasn't in the original df, then drop it!
    if imputation_strat == 'mean':
        if not output_name + '_mean' in orig:
            df = df.drop(output_name + '_mean', axis=1)

    return df


def add_rolling(df: pd.DataFrame,
                date_column: str,
                grouping_columns: list,
                value_column: str,
                windows=None,
                output_name=None,
                verbose = False,
                rolling_stat='mean'):
    """
    Function to add rolling average using all historical periods +
    a sliding window (defined by 'windows' optional argument)

    Args:
        df: Pandas dataframe
        date_column: Date column to use for rolling window
        grouping_columns: columns to group by, usually product/facility
        value_column: column to use to create mean
        rolling_stat: (rolling statistic); mean(avg), sum, std; default='mean'

    Optional arg:
        windows: other windows to use to create average, use by just listing
        (i.e. add_rolling(df, grouping_columns, date_column, quantity_column, 1, 2, 3, 4, 5)

    returns:
        df: pandas DataFrame w/rolling averages
    """

    if verbose:
        print("copying df..")

    df = df.copy()

    if output_name == None:
        output_name = value_column

    # first we need to ensure dataframe is actually grouped!
    if verbose:
        print("grouping columns..")

    df_ = df.groupby(grouping_columns + [date_column])[value_column].sum().reset_index()
    df_ = df_.set_index(date_column)

    # creating overall rolling mean
    # if column already exists pass
    if output_name + '_mean' not in df.columns:

        if verbose:
            print(f"calculating {rolling_stat}..")

        if rolling_stat == 'sum':
            tmp = df_.groupby(grouping_columns)[value_column].rolling(len(df), min_periods=1).sum().reset_index()
        elif rolling_stat == 'std':
            tmp = df_.groupby(grouping_columns)[value_column].rolling(len(df), min_periods=1).std().reset_index()
        elif rolling_stat == 'count':
            tmp = df_.groupby(grouping_columns)[value_column].rolling(len(df), min_periods=1).count().reset_index()
        else:
            tmp = df_.groupby(grouping_columns)[value_column].rolling(len(df), min_periods=1).mean().reset_index()

        tmp = tmp.rename({value_column: output_name + f'_{rolling_stat}'}, axis=1)

        if verbose:
            print("merging..")

        df = pd.merge(df, tmp, on=(grouping_columns + [date_column]), how='left')

    # looping through any additional windows to use to create rolling avg
    if windows:
        for i in windows:
            if verbose:
                print(f"running {i}...")

            # only continue if window does not already exist
            if output_name + f'_{rolling_stat}_' + str(i) not in df.columns:

                if rolling_stat == 'sum':
                    tmp = df_.groupby(grouping_columns)[value_column].rolling(i, min_periods=1).sum().reset_index()
                elif rolling_stat == 'std':
                    tmp = df_.groupby(grouping_columns)[value_column].rolling(i, min_periods=1).std().reset_index()
                    tmp = tmp.fillna(0)
                elif rolling_stat == 'count':
                    tmp = df_.groupby(grouping_columns)[value_column].rolling(i, min_periods=1).count().reset_index()
                else:
                    tmp = df_.groupby(grouping_columns)[value_column].rolling(i, min_periods=1).mean().reset_index()

                tmp = tmp.rename({value_column: output_name + f'_{rolling_stat}_' + str(i)}, axis=1)
                df = pd.merge(df, tmp, on=(grouping_columns + [date_column]), how='left')

    return df


def create_label(df: pd.DataFrame, date_column: str,
                 grouping_cols: list,
                 target_column: str,
                 lead_time=1,
                 frequency='M',
                 mode='train'):
    """
    Creates a new column called 'target' for prediction

    Args:
        df: Pandas DF of already aggregated/re-indexed time series
        date_column: date column for sorting
        grouping_cols: columns to group by, usually product/facility
        target_column: column to use to create target
        lead_time: number of periods to predict ahead

    Returns:
        df: pandas Dataframe but with target column
    """

    df = df.copy()

    # pre-sorting by date prior to shifting
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column)

    # grouping and shifting target column
    df['target'] = df.groupby(grouping_cols)[target_column].shift(-1 * lead_time)

    # next we need to get rid of any months that are outside of the lead time
    # (these values will be unknown but we need to distinguish them from 0s
    # from earlier months did not have reported utilization, i.e. = 0)
    if (mode == 'train') & (frequency == 'M'):
        cutoff = df[date_column].max() - relativedelta(months=lead_time - 1)
        df = df[df[date_column] < cutoff]

    # and now we can fill the rest of the missing months as 0
    # df['target'] = df['target'].fillna(0)
    df['target'] = df['target'].dropna()
    #df[target_column] = df[target_column].dropna()

    return df


def _deriv(df: pd.DataFrame, date_column: str,
           value_column: str, window: int):
    """
    First derivative of ts smoothed over window

    Args:
        df: Pandas dataframe
        date_column: Date column to use for rolling window
        value_column: Column to use to create derivative
        window: total periods to smooth over

    Returns:
        x: PD series of derivative
    """

    x = df.set_index(date_column)[value_column].bfill().rolling(window=window).mean().diff().fillna(0).reset_index()

    return x


def add_deriv(df: pd.DataFrame, date_column: str, grouping_cols: list,
              value_column: str, window: int):
    """
    Applies _deriv function to a TS dataframe

    Args:
        df: Pandas dataframe
        grouping_cols: columns to group by, usually product/facility
        date_column: Date column to use for rolling window
        value_column: Column to use to create derivative
        window: total periods to smooth over

    Returns:
        df: same Pandas DataFrame w/deriv column
    """

    # calculate temporary dataframe with derivative
    deriv_tmp = df.groupby(grouping_cols)[[value_column,
                                           date_column]].apply(lambda x: _deriv(x, date_column=date_column,
                                                                                value_column=value_column,
                                                                                window=window)).reset_index().rename(
        columns={value_column: 'first_deriv'}).drop('level_' + str(len(grouping_cols)), axis=1)

    # merge on grouping columns and date
    df = df.reset_index(drop=True).merge(deriv_tmp, on=grouping_cols + [date_column])

    return df

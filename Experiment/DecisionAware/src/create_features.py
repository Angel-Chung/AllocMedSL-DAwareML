"""wrapper to create features specific to essential meds forecasting"""

import sys; sys.path.append('..')
from conf.config import *
# add forecasting library path <-- should change after packaging the library
sys.path.append(FORECAST_LIB_DIR)
import pandas as pd
from retina.feature_engineering import (
    add_rolling,
    add_prev_periods,
    split_dates,
    str_to_cat,
    add_deriv,
    create_label)




def create_features_essential_meds(df: pd.DataFrame,
                                   grouping_cols=['fac_name', 'product'],
                                   date_column='date',
                                   lead_time=1,
                                   return_mapping=False) -> pd.DataFrame:
    """create features using forecasting library
    Args:
        df: PREPROCESSED dataframe
    Returns: df with ts features
    """

    df = df.copy()
    quantity_column = 'quantity'
    df = add_rolling(df, date_column, grouping_cols,
                         quantity_column, [2, 3, 4, 5, 6], rolling_stat='mean')
    df = add_prev_periods(df, date_column, grouping_cols,
                          quantity_column, 6)
    df = add_rolling(df, date_column, grouping_cols,
                         quantity_column, [3, 6], rolling_stat='std')
    df = split_dates(df, date_column)

    df = add_deriv(df, date_column, grouping_cols,
                   quantity_column, 3)
    df = add_rolling(df, date_column, ['product'],
                         quantity_column, [2, 3, 4, 5, 6, 10], output_name='avg_per_product')
    #df = df.fillna(0)  # fill standard deviation with 0 when mkissing

    if return_mapping:
        df, mapping = str_to_cat(df, return_mapping=True)
    else:
        df = str_to_cat(df)
    df = create_label(df, date_column, grouping_cols,
                      target_column=quantity_column,
                      lead_time=lead_time, mode='test')
    #adding to our date the total lead time!
    df['date'] = df.date + pd.DateOffset(months=lead_time)

    if return_mapping:
        return df, mapping
    return df

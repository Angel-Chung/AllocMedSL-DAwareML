# general
from conf.config import *
import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
import sys
sys.path.append('../')


def convert_raw_DHIS2_pivot(df, essential_meds):
    """clean and pivot the raw format of DHIS for a given list of essential meds """
    df_raw = df.copy()
    df_raw['date'] = pd.to_datetime(df_raw['Period'], format='%Y%m')
    # filter for essential meds
    for i in range(0, len(essential_meds)):
        if i == 0:
            df = df_raw[df_raw['Data_name'].str.contains(
                essential_meds[i], regex=False)]
            print(essential_meds[i], len(df))
        else:
            tmp = df_raw[df_raw['Data_name'].str.contains(
                essential_meds[i], regex=False)]
            df = pd.concat([df, tmp])
            print(essential_meds[i], len(tmp))

    # pivoting
    df['index'] = (df['date'].astype(str) + "_" + df['Organisation unit name'].astype(str) + "_" +
                   df['Organisation unit'].astype(str))

    df = df.pivot(index='index', columns="Data_name",
                  values="Value").reset_index()
    df['date'] = df['index'].astype(str).str.split(pat="_", expand=True)[0]
    df['fac_name'] = df['index'].astype(str).str.split(pat="_", expand=True)[1]
    df['fac_id'] = df['index'].astype(str).str.split(pat="_", expand=True)[2]
    return df


def add_sample_statistics_per_fac(df, num_nans=True, num_sample=True):
    """add number of nans of dispensed quantity"""

    df_pivot = df.copy()
    cols = df_pivot.columns[
        df_pivot.columns.str.contains('Dispensed')].tolist() + ['fac_name']

    df_sel = df_pivot[cols].set_index('fac_name')

    if num_nans:
        df_nans = df_sel.isna().groupby(['fac_name']).sum().sum(1)
        df_nans = pd.DataFrame(
            {'fac_name': df_nans.index.values, 'num_nans': df_nans.values})
    if num_sample:
        df_samples = df_sel.fillna(0).groupby(['fac_name']).count().mean(1)
        df_samples = pd.DataFrame(
            {'fac_name': df_samples.index.values, 'num_sample': df_samples.values})

    df_pivot = df_pivot.merge(df_nans, on='fac_name')
    df_pivot = df_pivot.merge(df_samples, on='fac_name')
    return df_pivot


def add_sample_statistics_per_fac_product(df, num_nans=True, num_sample=True):
    df_pivot = df.copy()
    cols_dis = df_pivot.columns[df_pivot.columns.str.contains(
        'Dispensed')].tolist()
    if num_nans:
        for i_cnt, i in enumerate(cols_dis):
            df_sel = df_pivot.set_index('fac_name')[i]
            if i_cnt == 0:
                df_nans = df_sel.isna().groupby(['fac_name']).sum()
                df_nans = pd.DataFrame(data={'num_nans_'+i[:-24]: df_nans.values},
                                       index=df_nans.index)
            else:
                df_tmp = df_sel.isna().groupby(['fac_name']).sum()
                df_nans.loc[df_tmp.index, 'num_nans_'+i[:-24]] = df_tmp.values
        print(df_nans.columns)
        df_pivot = df_pivot.merge(df_nans.reset_index(), on='fac_name')

    if num_sample:
        for i_cnt, i in enumerate(cols_dis):
            df_sel = df_pivot.set_index('fac_name')[i]
            if i_cnt == 0:
                df_samples = df_sel.fillna(0).groupby(['fac_name']).count()
                df_samples = pd.DataFrame(data={'num_sample_'+i[:-24]: df_samples.values},
                                          index=df_samples.index)
            else:
                df_tmp = df_sel.fillna(0).groupby(['fac_name']).count()
                df_samples.loc[df_tmp.index,
                               'num_sample_'+i[:-24]] = df_tmp.values
        print(df_samples.columns)
        df_pivot = df_pivot.merge(df_samples.reset_index(), on='fac_name')

    return df_pivot


def load_fromS3_convert_clean_save(path, s3_link, s3_link_geo, essential_meds, fac_type_list,
                                   save=True, fname='pivot_from_raw_DHIS2.csv',
                                   to_drop=['Unnamed: 0', 'Unnamed: 0.1',
                                            'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1'],
                                   use_existing_file_for_matching=True):
    """load raw DHIS2 from s3 and convert it to pivot table for a give essential meds """
    df_raw = pd.read_csv(s3_link, low_memory=False).drop(
        to_drop, axis=1).drop_duplicates().reset_index(drop=True)

    # filter for essential meds and convert raw data
    df_pivot = convert_raw_DHIS2_pivot(df_raw, essential_meds)
    geo_sel_cols = ['latitude', 'longitude', 'hf_pk']
    # match geo data to mfl
    if use_existing_file_for_matching:
        s3_l = 's3://sl-dashboard-data/raw/master_facility_list/mfl_to_dhis2_VR_GG.csv'
        df_geo = pd.read_csv(s3_l)
        df_geo = df_geo.rename(
            columns={'lat': 'latitude', 'long': 'longitude', 'id_dhis2': 'id'})
        mask = df_geo['manually_confirmed(Y/N)'] == 'Y'
        df_geo = df_geo[mask].reset_index(drop=True)
    else:
        # load geo coordinates
        df_geo = pd.read_csv(s3_link_geo)
        df_geo = add_hfpk_using_geo(df_geo, coor_col='coordinates')

    # add geo
    df_pivot['id'] = [i.split('_')[2] for i in df_pivot['index']]
    df_pivot = df_pivot.merge(df_geo[geo_sel_cols + ['id']], on='id')

    # drop columns
    df_pivot = df_pivot.drop(df_pivot.columns[
        df_pivot.columns.str.contains('Comments')].values, axis=1)

    # add facility type
    df_pivot['fac_type'] = df_pivot['fac_name'].apply(
        lambda x: x.split(' ')[-1])

    # data format
    df_pivot['date'] = pd.to_datetime(df_pivot['date'], errors='coerce')

    # include only selected fac types
    df_pivot = df_pivot[df_pivot['fac_type'].isin(
        fac_type_list)].reset_index(drop=True)

    # add sample statistics
    #df_pivot = add_sample_statistics_per_fac(df_pivot)
    df_pivot = add_sample_statistics_per_fac_product(df_pivot)
    # make necessary cols numeric
    for i in df_pivot.columns:
        if i not in ['index', 'date', 'fac_name', 'fac_id', 'fac_type', 'id']+geo_sel_cols:
            df_pivot[i] = pd.to_numeric(df_pivot[i], errors='coerce')

    # save
    if save:
        df_pivot.to_csv(path+fname, index=False)
    return df_pivot


def load_fromS3_merge_mfl_save(path, s3_link, s3_mfl, essential_meds, fac_type_list,
                               mfl_sel_cols=['lat', 'long', 'hf_pk'],
                               save=True, fname='pivot_from_raw_DHIS2.csv'):
    """load raw DHIS2 from s3 and convert it to pivot table for a give essential meds """
    df_raw = pd.read_csv(s3_link).drop(['Unnamed: 0', 'Unnamed: 0.1',
                                        'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1'],
                                       axis=1).drop_duplicates().reset_index(drop=True)

    # filter for essential meds and convert raw data
    df_pivot = convert_raw_DHIS2_pivot(df_raw, essential_meds)

    # load mfl
    df_mfl = pd.read_csv(s3_mfl)
    mask = df_mfl['manually_confirmed(Y/N)'] == 'Y'
    df_mfl = df_mfl[mask].reset_index(drop=True)

    # add geo
    df_pivot['id_dhis2'] = [i.split('_')[2] for i in df_pivot['index']]
    df_pivot = df_pivot.merge(
        df_mfl[mfl_sel_cols + ['id_dhis2']], on='id_dhis2')

    # drop columns
    df_pivot = df_pivot.drop(df_pivot.columns[
        df_pivot.columns.str.contains('Comments')].values, axis=1)

    # add facility type
    df_pivot['fac_type'] = df_pivot['fac_name'].apply(
        lambda x: x.split(' ')[-1])

    # data format
    df_pivot['date'] = pd.to_datetime(df_pivot['date'], errors='coerce')

    # include only selected fac types
    df_pivot = df_pivot[df_pivot['fac_type'].isin(
        fac_type_list)].reset_index(drop=True)

    # add sample statistics
    #df_pivot = add_sample_statistics_per_fac(df_pivot)
    df_pivot = add_sample_statistics_per_fac_product(df_pivot)
    # make necessary cols numeric
    for i in df_pivot.columns:
        if i not in ['index', 'date', 'fac_name', 'fac_id', 'fac_type', 'id_dhis2']+mfl_sel_cols:
            df_pivot[i] = pd.to_numeric(df_pivot[i], errors='coerce')

    # save
    if save:
        df_pivot.to_csv(path+fname, index=False)
    return df_pivot


def sep_factype_from_facname(df, fac_type_list, col='facility_name', return_factype=True,
                             return_facname=True, prefix='seperated_'):
    """seperate facility type from facility name"""
    df_ = df.copy()
    df_[prefix+col] = df_[col].apply(lambda x: ' '.join(x.split(
        ' ')[:-1]) if x.split(' ')[-1] in fac_type_list else x)

    df_[prefix+'fac_type'] = df_[col].apply(lambda x: x.split(
        ' ')[-1] if x.split(' ')[-1] in fac_type_list else np.nan)

    return df_


def add_hfpk_using_geo(df, lat_col='latitude', long_col='longitude',
                       score_thr=0.99, coor_col=None, filter_point_coor=True,
                       s3_mfl_link="s3://sl-dashboard-data/normalized/master_facility_list/master_facility_update_9.csv"):
    """add hf_pk from master facility list to  the given dataframe using geo location """
    import recordlinkage as rl

    df_geo = df.copy()
    df_mfl = pd.read_csv(s3_mfl_link)

    # only indlude point coordinates
    if filter_point_coor:
        mask = df_geo.featureType == 'POINT'
        df_geo = df_geo[mask].reset_index(drop=True)

    if coor_col != None:
        df_geo[lat_col] = [
            eval(i)[1] if i == i else np.nan for i in df_geo[coor_col].values]
        df_geo[long_col] = [
            eval(i)[0] if i == i else np.nan for i in df_geo[coor_col].values]

    # Create an indexer object
    indexer = rl.Index()
    indexer.full()

    # Create candidate pairs
    pairs = indexer.index(df_geo, df_mfl)

    # Creat a comparion object
    compare = rl.Compare()
    # GEO
    compare.geo(left_on_lat=lat_col, left_on_lng=long_col,
                right_on_lat='lat', right_on_lng='long',
                method='squared', label='score')

    matches = compare.compute(pairs, df_geo, df_mfl)

    # threshold for high score
    ma = matches.reset_index()
    ma_uniq_left = ma.loc[ma.groupby('level_0')['score'].idxmax().values]
    mask1 = ma_uniq_left.score > score_thr
    ma_sel = ma_uniq_left[mask1]

    idx_right = ma_sel.groupby('level_1')['score'].idxmax().values
    final_match = ma_sel.loc[idx_right]

    sel_col_mfl = ['hf_pk']
    df_geo_matched = pd.concat([df_geo.loc[final_match['level_0']].reset_index(drop=True),
                                df_mfl[sel_col_mfl].loc[final_match['level_1']].reset_index(drop=True)], axis=1)

    return df_geo_matched


def deal_with_zeors_nans_inDHIS2(df, product_names,
                                 ops=['invalid_zeros',
                                      'stock_sum_err',
                                      'drop_stockout']):
    """deal with zeros and nans in DHIS2 data with different strategies.
    selected rows will turn to nans for dispensed vals.

    invalid_zeros: All stock info (possibly with the exception of stock_ordered) 
    are recorded as '0' for a given contraceptive product, month and service delivery site

    stock_sum_err: relations between stockout, Closing/Opening Balance, Received and
    dispensed doesnt add up

    drop_stockout: drop month with stockout as the dispensed values are influenced
    """
    df_pivot = df.copy()

    if 'invalid_zeros' in ops:
        sel_stock_abbr = ['- Quantity Dispensed (D)',
                          '- Closing Balance (E)',
                          '- Days Out of Stock (F)',
                          '- Losses / Adjustments (C)',
                          '- Opening Balance (A)',
                          '- Quantity Received (B)']
        for i in product_names:
            sel_col = [i+j for j in sel_stock_abbr]
            mask = df_pivot[sel_col].sum(1) == 0
            df_pivot.loc[mask.values, sel_col[0]] = np.nan

    if 'stock_sum_err' in ops:
        for i in product_names:
            A = df_pivot[i + '- Opening Balance (A)']
            B = df_pivot[i + '- Quantity Received (B)']
            C = df_pivot[i + '- Losses / Adjustments (C)']
            D = df_pivot[i + '- Quantity Dispensed (D)']
            E = df_pivot[i + '- Closing Balance (E)']

            # check where opening balance + received and closing balance are reported correctly:
            # (A) + (B) - (E) = (D) ??
            ApBmE = A + B - E
            mask1 = ApBmE == D

            # adjusment --> find rows that recieved + adjusted == dispensed + Closing balance
            BpC = B + C
            DpE = D + E
            mask2 = BpC == DpE
            mask = mask1 | mask2
            df_pivot.loc[~mask.values, i + '- Quantity Dispensed (D)'] = np.nan

    if 'drop_stockout' in ops:
        # remove (make nans) rows where stockout is positive

        sel_stock_abbr = ['- Quantity Dispensed (D)',
                          '- Closing Balance (E)',
                          '- Days Out of Stock (F)',
                          '- Losses / Adjustments (C)',
                          '- Opening Balance (A)',
                          '- Quantity Received (B)']

        for i in product_names:
            sel_col = i + '- Days Out of Stock (F)'
            mask = df_pivot[sel_col] > 0
            df_pivot.loc[mask.values, i+'- Quantity Dispensed (D)'] = np.nan

    return df_pivot


def add_cols_from_mfl_on_hfpk(df, cols=['region', 'chiefdom', 'district'],
                              s3_mfl_link="s3://sl-dashboard-data/normalized/master_facility_list/master_facility_update_9.csv"):
    """merge cols from master health facilities to a given datafram"""
    df_master = pd.read_csv(s3_mfl_link)
    return df.merge(df_master[['hf_pk']+cols], on='hf_pk')


def create_unpivot_dataframe(df_pivot, id_vars, key_word_valvars,
                             var_name, val_name):

    val_vars = df_pivot.columns[
        df_pivot.columns.str.contains(key_word_valvars, regex=False)].tolist()
    df = df_pivot.melt(id_vars=id_vars, var_name=var_name,
                       value_vars=val_vars, value_name=val_name)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df


def melt_multiplecols(df_pivot, id_vars, key_word_valvars,
                      var_name, val_name, num_cols, product_names):

    if len(id_vars) != num_cols or len(key_word_valvars) != num_cols:
        raise ValueError("all input should be the same length")
    if len(var_name) != num_cols or len(val_name) != num_cols:
        raise ValueError("all input should be the same length")

    for k in range(num_cols):
        df = create_unpivot_dataframe(df_pivot, id_vars[k], key_word_valvars[k],
                                      var_name[k], val_name[k])

        for p in product_names:
            df.loc[df['product'].str.contains(p, regex=False),
                   'product'] = p

        if k == 0:
            df_final = df.copy()
        else:
            df_final = pd.merge(
                df_final, df, on=['fac_name', 'product', 'date'])

    return df_final


def extract_monthlyavg_fromDHIS2(df, m=3):
    """extract monthly average values from DHIS2"""
    sel_cols = ['date', 'fac_name'] +\
        list(df.columns[df.columns.str.contains(
            str(m)+' Months', regex=False)].values)
    df_monthavg = df[sel_cols].melt(
        ['date', 'fac_name'],
        value_name='avg_'+str(m)+'month_DHIS2').rename(columns={'variable': 'product'})

    df_monthavg['date'] = pd.to_datetime(
        df_monthavg['date'], errors='coerce')

    # change product name for a new column
    df_monthavg['product'] = df_monthavg['product'].str.replace(
        'Average Monthly Consumption for ', '')
    df_monthavg['product'] = df_monthavg['product'].str.replace(
        ' - '+str(m)+' Months', ' - Quantity Dispensed (D)')
    return df_monthavg


def apply_ARIMA(df, test_start_date, max_date, freq='MS', order_pqd=(3, 1, 1)):
    """Apply arima on a given time series"""
    from statsmodels.tsa.arima.model import ARIMA
    # define time steps of the ts in two weeks frequency
    df = df[['date', 'quantity']]
    df_org = df.copy()
    da = pd.date_range(df['date'].min(), max_date,
                       freq=freq)

    df = df.rename(columns={'quantity': 'y_arima'})

    # num samples
    len_df = len(df)

    # reindex df to include all dates
    df = df.reset_index(drop=True).set_index(
        'date').reindex(da).reset_index().rename(columns={'index': 'date'})
    # train only on the training set
    mask = df.date < test_start_date
    df_train = df[mask]

    # apply ARIMA
    if len_df > 5:
        try:
            df_train.y_arima.interpolate(method='polynomial',
                                         order=1, inplace=True)
            df_train.fillna(df.y_arima.median(), inplace=True)
            model = ARIMA(df_train.y_arima,
                          dates=df_train.date,
                          freq=freq, order=order_pqd)
            model_fit = model.fit()
            new = model_fit.predict(end=len(df))
            df.loc[range(len(df)), 'y_arima'] = new.values[1:]
        except:
            df.loc[range(len(df)), 'y_arima'] = df_train.y_arima.mean(
                skipna=True)
    else:
        df.loc[range(len(df)), 'y_arima'] = df_train.y_arima.mean(skipna=True)
    df_final = df.set_index('date').loc[df_org['date']]
    return df_final.fillna(df_final.median())


def add_deriv(df, window=3):
    """first derivative of ts smoothed over wíndow"""
    x = df.set_index('date').quantity.shift(1).bfill().rolling(
        window=window).mean().diff().fillna(0).reset_index()
    return x


def split_dates(df, date_column):

    date_df = pd.DataFrame({"year": df[date_column].dt.year,
                            "month": df[date_column].dt.month,
                            # "day": df[date_column].dt.day,
                            "dayofweek": df[date_column].dt.dayofweek,
                            # "dayofyear": df[date_column].dt.dayofyear,
                            # "weekofyear": df[date_column].dt.isocalendar().week,
                            # "quarter": df[date_column].dt.quarter,
                            })

    # add new date features to dataframe
    df = pd.concat([df, date_df], axis=1)
    return df


def str_to_cat(df, output_file=None):
    from pandas.api.types import is_string_dtype, is_numeric_dtype
    for p, q in df.items():
        if is_string_dtype(q):
            df[p] = q.astype('category').cat.as_ordered()
            tmp_dict = dict(enumerate(df[p].cat.categories))

    return df


def basic_faetures_ts(df, max_date, test_start_date,
                      ops=['derivative', 'prev_months',
                           'cat_to_num', 'split_dates', 'cat_to_oneHot'],
                      cats=['product', 'fac_name', 'fac_type']):
    """add basic feature eng for time series"""
    df2 = df.copy()
    if 'derivative' in ops:
        print('add derivatives ...')
        df_deriv = df2.groupby(['fac_name', 'product'])[[
            'quantity', 'date']].apply(add_deriv).reset_index().rename(
                columns={'quantity': 'first_deriv'}).drop('level_2', axis=1)
        df2 = df2.reset_index(drop=True).merge(df_deriv, on=[
            'fac_name', 'product', 'date'])

    if 'prev_months' in ops:
        print('add previous months and 3month average ...')
        pred_period = int((max_date - test_start_date).days/30) + 1
        num_month = 3  # number of months to be included
        for i in range(num_month):
            df_prev_temp = df2.groupby(['fac_name', 'product'])[
                'quantity'].transform(lambda x: x.shift(
                    i+1, fill_value=x[:-pred_period].median(skipna=True)))

            df2['prev_month_'+str(i+1)] = list(df_prev_temp.fillna(0).values)

        # 3 month rolling average
        df2['3MonthAvg'] = df2[['prev_month_1', 'prev_month_2',
                                'prev_month_3']].mean(1, skipna=True)

    if 'cat_to_num' in ops:
        print('add cat to num ...')
        df2 = str_to_cat(df2).reset_index(drop=True)
        for i in cats:
            df2[i+'_id'] = df2[i].cat.codes
        #df2['fac_name_id'] = df2['fac_name'].cat.codes
        #df2['fac_type_id'] = df2['fac_type'].cat.codes
        df2 = df2.reset_index(drop=True)

    if 'cat_to_oneHot' in ops:
        df2 = str_to_cat(df2).reset_index(drop=True)
        df2 = pd.concat([df2, pd.get_dummies(
            df2['product'], prefix='product')], 1)

    if 'split_dates' in ops:
        df2['date'] = pd.to_datetime(df2['date'], errors='coerce')
        df2 = split_dates(df2, 'date')

    if "add_avg_per_product" in ops:
        print("add_avg_per_product ...")
        df_train = df2[df2['date'] < test_start_date]
        avg_prod = df_train.groupby('product')[
            ['quantity']].mean().reset_index().rename(
                columns={'quantity': 'avg_quantity_per_product'})
        df2 = df2.merge(avg_prod, on='product')

    if "add_avg_per_fac_product" in ops:
        print("add_avg_per_fac_product ...")
        df_train = df2[df2['date'] < test_start_date]
        avg_prod = df_train.groupby(['fac_name', 'product'])[
            ['quantity']].mean().reset_index().rename(
                columns={'quantity': 'avg_quantity_per_fac_product'})
        df2 = df2.merge(avg_prod, on=['fac_name', 'product'])

    if "add_std_per_fac_product" in ops:
        print("add_std_per_fac_product ...")
        df_train = df2[df2['date'] < test_start_date]
        avg_prod = df_train.groupby(['fac_name', 'product'])[
            ['quantity']].std().reset_index().rename(
                columns={'quantity': 'std_quantity_per_fac_product'})
        df2 = df2.merge(avg_prod, on=['fac_name', 'product'])

    if "add_count_per_fac_product" in ops:
        print("add_count_per_fac_product ...")
        df_train = df2[df2['date'] < test_start_date]
        avg_prod = df_train.groupby(['fac_name', 'product'])[
            ['quantity']].count().reset_index().rename(
                columns={'quantity': 'count_quantity_per_fac_product'})
        df2 = df2.merge(avg_prod, on=['fac_name', 'product'])
    return df2


def impute_missing(df, test_start_date, groups=['fac_name', 'product'],
                   target='quantity', strategy='median', min_sample=10):
    """impute missing values"""
    df_ = df.copy()
    mask = df_['date'] < test_start_date
    df_train = df_[mask]

    if strategy == 'median':
        print("impute using median ...")
        imputed_target = df_train.groupby(groups)[target].apply(
            lambda x: x.fillna(x.median()) if len(x[~x.isna()]) > min_sample else x)
        df_.loc[mask, 'quantity'] = imputed_target.values

    if strategy == 'interpolate':
        print("impute using linear interpolation ...")
        imputed_target = df_train.groupby(groups)[target].apply(lambda x: x.interpolate(
            method='linear', order=3) if len(x[~x.isna()]) > min_sample else x)
        df_.loc[mask, 'quantity'] = imputed_target.values

    return df_


def advanced_feature_ts(df, path, max_date, test_start_date,
                        ops=['ARIMA', 'ts_statistics'], redo=False):
    """calculate advanced feature for time series"""
    df_ = df.copy()

    if 'ARIMA' in ops:
        from statsmodels.tsa.arima.model import ARIMA
        print('add ARIMA features ...')
        df_['date'] = pd.to_datetime(df_['date'], errors='coerce')
        df_ = df_.sort_values(['date']).reset_index(drop=True)

        # check if arima extraction already exist or not
        if os.path.isfile(path + 'df_arima.csv') and redo == False:
            df_arima = pd.read_csv(path + 'df_arima.csv')
            df_arima['date'] = pd.to_datetime(df_arima['date'],
                                              errors='coerce')
        else:
            df_arima = df_.groupby(
                ['fac_name', 'product'])[['quantity', 'date']].parallel_apply(
                    apply_ARIMA, test_start_date=test_start_date,
                    max_date=max_date,
                    order_pqd=(3, 1, 1)).reset_index()
            df_arima.to_csv(path + 'df_arima.csv', index=False)

        df_ = pd.merge(df_,
                       df_arima,
                       on=['fac_name', 'product', 'date'])

    if 'ts_statistics' in ops:
        print('add time series statistics using tsfresh ...')
        from tsfresh.feature_extraction import MinimalFCParameters, extract_features
        df_ = df_.sort_values(['date']).reset_index(drop=True)
        mask = df_['date'] < test_start_date
        df_train = df_[mask]
        # check if extraction already exist or not
        if os.path.isfile(path + 'df_extracted.csv') and redo == False:
            df_extracted = pd.read_csv(path + 'df_extracted.csv')
        else:
            df_extracted = df_train.reset_index(drop=True).groupby(['fac_name'])[[
                'quantity', 'date', 'product']].apply(
                    extract_features, default_fc_parameters=MinimalFCParameters(),
                    column_id='product', column_sort='date')
            df_extracted = df_extracted.reset_index().rename(
                columns={'level_1': 'product'})
            df_extracted.to_csv(path + 'df_extracted.csv', index=False)

        sel_fea = df_extracted.columns
        df_ = df_.reset_index(drop=True).merge(
            df_extracted[sel_fea].dropna(axis=1), on=['fac_name', 'product'])

    return df_


def MAPE_per_product(y_pred, y_true, avg_per_product):
    """calc MAPE but average over mean of each product"""
    return np.mean(np.abs(y_pred - y_true)/avg_per_product)


def _MAPE(y1, y2, ignore_zeros=True):
    y1 = np.array(y1)
    y2 = np.array(y2)
    if ignore_zeros:
        mask = y2 != 0
        y2 = y2[mask]
        y1 = y1[mask]
        y_true = y2
    else:
        y_true = y2.copy()
        mask = y2 == 0
        y_true[mask] = 1.
    return np.array(np.abs(y1-y2)/np.abs(y_true))


def MAPE(y1, y2, target_integer=True):
    if target_integer and (np.int64(y2) - y2).sum() != 0:
        raise ValueError('Check the order of prediction and true values!')
    return np.median(_MAPE(y1, y2))


def rsq_n_rmse_MAPE(model, x_train, y_train, x_val, y_val):
    from sklearn.metrics import r2_score, mean_squared_error
    result = {}
    if model == None:
        pred_train = x_train
        pred_val = x_val
    else:
        pred_train = model.predict(x_train)
        pred_val = model.predict(x_val)

    result = {}

    # r2 score
    result['train rsq'] = r2_score(y_train, pred_train)
    result['val rsq'] = r2_score(y_val, pred_val)

    # rmse
    result['train rmse'] = np.sqrt(mean_squared_error(pred_train, y_train))
    result['val rmse'] = np.sqrt(mean_squared_error(pred_val, y_val))

    # MAPE
    result['train MAPE'] = MAPE(pred_train, y_train)
    result['val MAPE'] = MAPE(pred_val, y_val)

    return result


def weighted_MAPE(ypred, ytrue, neg2posCoeff=0.2, ignore_zeros=True):
    y1 = np.array(ypred)
    y2 = np.array(ytrue)
    if ignore_zeros:
        mask = y2 != 0
        y2 = y2[mask]
        y1 = y1[mask]
        y_true = y2
    else:
        y_true = y2.copy()
        mask = y2 == 0
        y_true[mask] = 1.
    err = y1-y2
    mask = err < 0
    y1[mask] *= neg2posCoeff
    return np.median(np.abs(y1-y2)/y2)


def asymm_err(y1, y2):
    """calculates number of over/under-predictions and mean percentage error for each"""
    err_all = (y1 - y2)/y2
    mask_pos = err_all > 0
    mask_neg = err_all < 0
    num_pos_err = np.sum(mask_pos)
    num_neg_err = np.sum(mask_neg)

    MAPE_pos = np.median(err_all[mask_pos])
    MAPE_neg = np.median(err_all[mask_neg])
    return num_pos_err, num_neg_err, MAPE_pos, MAPE_neg


def oversample(X_train, Y_train, perc=75, repetition=1):
    """oversampling by repeating selected rows in the training set"""

    thr = np.percentile(Y_train, perc)
    mask = Y_train >= thr

    X_train_oversample = X_train.copy()
    Y_train_oversample = Y_train.copy()
    for i in range(repetition):
        X_train_oversample = pd.concat(
            [X_train_oversample, X_train[mask]]).reset_index(drop=True)
        Y_train_oversample = pd.concat(
            [Y_train_oversample, Y_train[mask]]).reset_index(drop=True)
    return X_train_oversample, Y_train_oversample


def oversample_per_cat(X_train, Y_train, perc=75, repetition=1, cat_label='product_id'):

    X_train_oversample = X_train.copy()
    Y_train_oversample = Y_train.copy()
    for p in X_train[cat_label].unique():
        mask = X_train[cat_label] == p
        Y_train_sel = Y_train[mask]
        X_train_sel = X_train[mask]
        thr = np.percentile(Y_train_sel, perc)
        mask1 = Y_train_sel >= thr
        for i in range(repetition):
            X_train_oversample = pd.concat(
                [X_train_oversample, X_train_sel[mask1]]).reset_index(drop=True)
            Y_train_oversample = pd.concat(
                [Y_train_oversample, Y_train_sel[mask1]]).reset_index(drop=True)
    return X_train_oversample, Y_train_oversample


def calc_err_per_cat(preds, X_test, Y_test, thr=0,
                     cat='product_id', cat_ids=[6, 1, 5, 7, 0, 2, 3, 4]):
    """calculates MAPE per product for baseline and model"""
    mask = Y_test >= thr
    Y_test_sel, X_test_sel, preds_sel = Y_test[mask], X_test[mask], preds[mask]
    mape_lst = []
    mape_avg = []

    prod_abbr = []
    prod_id_orders = cat_ids
    for i in prod_id_orders:
        mask = X_test_sel[cat] == i
        mape_lst.append(MAPE(preds_sel[mask], Y_test_sel[mask]))
        mape_avg.append(
            MAPE(X_test_sel['avg_3months_DHIS2'][mask], Y_test_sel[mask]))
    return mape_lst, mape_avg, cat_ids


def load_clean_prepare_4ML(df_pivot):
    """clean, preprocess and make the raw DHIS2 dat ready for ML"""
    df_pivot = add_cols_from_mfl_on_hfpk(
        df_pivot, cols=['region', 'chiefdom', 'district'])
    # deal with zeros
    df_pivot = deal_with_zeors_nans_inDHIS2(df_pivot, product_names,
                                            ops=['invalid_zeros',
                                                 'stock_sum_err',
                                                 'drop_stockout'])
    # melt pivot table and create dataframe
    num_cols = 5
    id_vars = [['date', 'fac_name', 'fac_type', 'latitude',
                'longitude', 'hf_pk', 'region', 'chiefdom', 'district']] +\
        [['date', 'fac_name']]*(num_cols-1)
    key_word_valvars = ['Dispensed', '3 Months',
                        '6 Months', 'num_nans', 'num_sample']
    var_name = ['product']*num_cols
    val_name = ['quantity', 'avg_3months_DHIS2',
                'avg_6months_DHIS2', 'num_nans', 'num_sample']

    df = melt_multiplecols(df_pivot, id_vars, key_word_valvars,
                           var_name, val_name, num_cols, product_names)
    # drop NANs in train period
    df = df.sort_values(['date'])
    df = df.dropna(subset=['quantity'])
    df = df.fillna(0).reset_index(drop=True)

    # remove values above threshold
    df_nonzero = df[df.quantity != 0]
    df_thr = df_nonzero.groupby(['product', 'fac_type'])[
        'quantity'].quantile(0.99)
    print(df_thr)
    for i in df_thr.index:
        mask = df['product'] == i[0]
        mask1 = df['fac_type'] == i[1]
        mask2 = df.quantity > df_thr[i]
        idx_to_remove = df[mask & mask1 & mask2].index
        df = df.drop(index=idx_to_remove).reset_index(drop=True)
    # add meds category
    df['meds_category'] = np.nan
    for k in product_group_idx.keys():
        sel_prods = [product_names[i] for i in product_group_idx[k]]
        mask = df['product'].isin(sel_prods)
        df.loc[mask, 'meds_category'] = k
    return df


def run_rf(df, test_date, target, date_col, params, group_cols=None, agg_district=False):
    """split train/test and run random forest regressor"""
    df = df.reset_index(drop=True).copy()
    test_date = pd.to_datetime(test_date)
    if agg_district:
        df = df.groupby(group_cols).sum().reset_index()
    df_train = df[df[date_col] < test_date]
    df_test = df[df[date_col] == test_date]

    Y_train = df_train[target]
    X_train = df_train.drop([target, date_col], axis=1)

    X_test = df_test.drop([target, date_col], axis=1)
    Y_test = df_test[target]

    rf_regressor = RandomForestRegressor(n_jobs=-1, **params)
    rf_regressor.fit(X_train, Y_train)
    return rf_regressor, df_train, df_test


def run_ML(df, test_date, target, date_col, params, model='rf', group_cols=None, agg_district=False):
    """split train/test and run ML regressor"""
    df = df.reset_index(drop=True).copy()
    test_date = pd.to_datetime(test_date)
    if agg_district:
        df = df.groupby(group_cols).sum().reset_index()
    df_train = df[df[date_col] < test_date]
    df_test = df[df[date_col] == test_date]

    Y_train = df_train[target]
    X_train = df_train.drop([target, date_col], axis=1)

    X_test = df_test.drop([target, date_col], axis=1)
    Y_test = df_test[target]

    if model == 'rf':
        regressor = RandomForestRegressor(n_jobs=-1, **params)
    elif model == 'lgb':
        regressor = lgb.LGBMRegressor(**params)
    else:
        NotImplemented
    regressor.fit(X_train, Y_train)
    return regressor, df_train, df_test


def map_learned_trees(trained_model, df_train, df_test, test_date, model='rf', num_month_valid=3):
    """train extra model to map trees output to the target"""
    val_date = [pd.to_datetime(test_date) - pd.DateOffset(months=i+1)
                for i in range(num_month_valid)]
    X_test = df_test.drop(['quantity', 'date'], axis=1)
    df_valid = df_train[df_train.date.isin(val_date)]
    X_valid = df_valid.drop(['quantity', 'date'], axis=1)
    pred_per_tree = np.stack([t.predict(X_valid.to_numpy())
                             for t in trained_model.estimators_])
    if model == 'rf':
        m_reg = RandomForestRegressor(n_jobs=-1)
    elif model == 'LR':
        m_reg = Ridge(alpha=0.2, normalize=False)
    elif model == 'lgb':
        m_reg = lgb.LGBMRegressor()

    m_reg.fit(pred_per_tree.T, df_valid.quantity.values)
    # test
    pred_per_tree_test = np.stack(
        [t.predict(X_test.to_numpy()) for t in trained_model.estimators_])
    pred_multi_learning = m_reg.predict(pred_per_tree_test.T)
    return pred_multi_learning

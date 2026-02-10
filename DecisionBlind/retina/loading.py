import pandas as pd
import numpy as np
from datetime import datetime
import os
import recordlinkage


def convert_raw_DHIS2_pivot(df, essential_meds):
    """
    Clean and pivot the raw format of DHIS for a given list of essential meds
    """

    df_raw = df.copy()
    df_raw['date'] = pd.to_datetime(df_raw['Period'], format='%Y%m')
    # filter for essential meds
    for i in range(0, len(essential_meds)):
        if i == 0:
            df = df_raw[df_raw['Data_name'].str.contains(essential_meds[i], regex=False)]
            print(essential_meds[i], len(df))
        else:
            tmp = df_raw[df_raw['Data_name'].str.contains(essential_meds[i], regex=False)]
            df = pd.concat([df, tmp])
            print(essential_meds[i], len(tmp))

    # pivoting
    df['index'] = (df['date'].astype(str) + "_" + df['Organisation unit name'].astype(str) + "_" +
                   df['Organisation unit'].astype(str))

    df = df.pivot(index='index', columns="Data_name", values="Value").reset_index()
    df['date'] = df['index'].astype(str).str.split(pat="_", expand=True)[0]
    df['fac_name'] = df['index'].astype(str).str.split(pat="_", expand=True)[1]
    df['fac_id'] = df['index'].astype(str).str.split(pat="_", expand=True)[2]

    return df


def add_sample_statistics_per_fac(df, num_nans=True, num_sample=True):
    """
    Add number of nans of dispensed quantity
    """

    df_pivot = df.copy()
    cols = df_pivot.columns[df_pivot.columns.str.contains('Dispensed')].tolist() + ['fac_name']

    df_sel = df_pivot[cols].set_index('fac_name')

    if num_nans:
        df_nans = df_sel.isna().groupby(['fac_name']).sum().sum(1)
        df_nans = pd.DataFrame({'fac_name': df_nans.index.values, 'num_nans': df_nans.values})
    if num_sample:
        df_samples = df_sel.fillna(0).groupby(['fac_name']).count().mean(1)
        df_samples = pd.DataFrame({'fac_name': df_samples.index.values, 'num_sample': df_samples.values})

    df_pivot = df_pivot.merge(df_nans, on='fac_name')
    df_pivot = df_pivot.merge(df_samples, on='fac_name')

    return df_pivot


def add_sample_statistics_per_fac_product(df, num_nans=True, num_sample=True):

    df_pivot = df.copy()
    cols_dis = df_pivot.columns[df_pivot.columns.str.contains('Dispensed')].tolist()

    if num_nans:
        for i_cnt, i in enumerate(cols_dis):
            df_sel = df_pivot.set_index('fac_name')[i]
            if i_cnt == 0:
                df_nans = df_sel.isna().groupby(['fac_name']).sum()
                df_nans = pd.DataFrame(data={'num_nans_' + i[:-24]: df_nans.values}, index=df_nans.index)
            else:
                df_tmp = df_sel.isna().groupby(['fac_name']).sum()
                df_nans.loc[df_tmp.index, 'num_nans_' + i[:-24]] = df_tmp.values

        print(df_nans.columns)
        df_pivot = df_pivot.merge(df_nans.reset_index(), on='fac_name')

    if num_sample:
        for i_cnt, i in enumerate(cols_dis):
            df_sel = df_pivot.set_index('fac_name')[i]
            if i_cnt == 0:
                df_samples = df_sel.fillna(0).groupby(['fac_name']).count()
                df_samples = pd.DataFrame(data={'num_sample_' + i[:-24]: df_samples.values},
                                          index=df_samples.index)
            else:
                df_tmp = df_sel.fillna(0).groupby(['fac_name']).count()
                df_samples.loc[df_tmp.index, 'num_sample_' + i[:-24]] = df_tmp.values

        print(df_samples.columns)
        df_pivot = df_pivot.merge(df_samples.reset_index(), on='fac_name')

    return df_pivot


def load_fromS3_convert_clean_save(path, s3_link, s3_link_geo, essential_meds, fac_type_list,
                                   save=True, fname='pivot_from_raw_DHIS2.csv',
                                   to_drop=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1'],
                                   use_existing_file_for_matching=True):
    """
    Load raw DHIS2 from s3 and convert it to pivot table for a give essential meds
    """

    df_raw = pd.read_csv(s3_link, low_memory=False)
    to_drop = df_raw.columns[df_raw.columns.isin(to_drop)]
    df_raw = df_raw.drop(to_drop, axis=1).drop_duplicates().reset_index(drop=True)

    ########## HACKY SOLUTION ###########
    # in changing data loader to DB, at the moment, April 5th 2022, there are
    # intermediate csv file generated. The following line tries to make the old
    # loading compatible with the new data. Everything should be rewritten when the
    # DB is up and running
    if "Data_name" not in df_raw.columns:
        fn_dataelem = "data/AWS/raw/dataelement_meta.csv"
        fn_org_unit = "data/AWS/raw/organisationunit_meta.csv"
        df_dataemel = pd.read_csv(fn_dataelem)
        df_orgunit = pd.read_csv(fn_org_unit)
        df_raw = df_raw.merge(df_dataemel[['dataelement_id', 'name']], on='dataelement_id').rename(
            columns={'name': 'Data_name'})

        df_raw = df_raw.merge(df_orgunit[['organisationunit_id', 'name']], on='organisationunit_id').rename(
            columns={'name': 'Organisation unit name'})
        dict_name = {'period': 'Period', 'data': 'Value', 'organisationunit_id': 'Organisation unit'}
        df_raw = df_raw.drop(['month', 'year', 'dataelement_id'], axis=1).rename(columns=dict_name)

    # filter for essential meds and convert raw data
    df_pivot = convert_raw_DHIS2_pivot(df_raw, essential_meds)
    geo_sel_cols = ['latitude', 'longitude', 'hf_pk']
    # match geo data to mfl
    if use_existing_file_for_matching:
        s3_l = 'data/AWS/raw/mfl_to_dhis2_VR_GG.csv'
        df_geo = pd.read_csv(s3_l)
        df_geo = df_geo.rename(columns={'lat': 'latitude', 'long': 'longitude', 'id_dhis2': 'id'})
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
    df_pivot = df_pivot.drop(df_pivot.columns[df_pivot.columns.str.contains('Comments')].values, axis=1)

    # add facility type
    df_pivot['fac_type'] = df_pivot['fac_name'].apply(lambda x: x.split(' ')[-1])

    # data format
    df_pivot['date'] = pd.to_datetime(df_pivot['date'], errors='coerce')

    # include only selected fac types
    df_pivot = df_pivot[df_pivot['fac_type'].isin(fac_type_list)].reset_index(drop=True)

    # add sample statistics
    # df_pivot = add_sample_statistics_per_fac(df_pivot)
    df_pivot = add_sample_statistics_per_fac_product(df_pivot)
    # make necessary cols numeric
    for i in df_pivot.columns:
        if i not in ['index', 'date', 'fac_name', 'fac_id', 'fac_type', 'id'] + geo_sel_cols:
            df_pivot[i] = pd.to_numeric(df_pivot[i], errors='coerce')

    # save
    if save:
        df_pivot.to_csv(os.path.join(path, fname), index=False)

    return df_pivot


def load_fromS3_merge_mfl_save(path, s3_link, s3_mfl, essential_meds, fac_type_list,
                               mfl_sel_cols=['lat', 'long', 'hf_pk', 'chiefdom', 'district', 'region'],
                               save=True, fname='pivot_from_raw_DHIS2.csv'):
    """
    Load raw DHIS2 from s3 and convert it to pivot table for a give essential meds
    """
    df_raw = pd.read_csv(s3_link).drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1'],
                                       axis=1).drop_duplicates().reset_index(drop=True)

    # filter for essential meds and convert raw data
    df_pivot = convert_raw_DHIS2_pivot(df_raw, essential_meds)

    # load mfl
    df_mfl = pd.read_csv(s3_mfl)
    mask = df_mfl['manually_confirmed(Y/N)'] == 'Y'
    df_mfl = df_mfl[mask].reset_index(drop=True)

    # add geo
    df_pivot['id_dhis2'] = [i.split('_')[2] for i in df_pivot['index']]
    df_pivot = df_pivot.merge(df_mfl[mfl_sel_cols + ['id_dhis2']], on='id_dhis2')

    # drop columns
    df_pivot = df_pivot.drop(df_pivot.columns[df_pivot.columns.str.contains('Comments')].values, axis=1)

    # add facility type
    df_pivot['fac_type'] = df_pivot['fac_name'].apply(lambda x: x.split(' ')[-1])

    # data format
    df_pivot['date'] = pd.to_datetime(df_pivot['date'], errors='coerce')

    # include only selected fac types
    df_pivot = df_pivot[df_pivot['fac_type'].isin(fac_type_list)].reset_index(drop=True)

    # add sample statistics
    # df_pivot = add_sample_statistics_per_fac(df_pivot)
    df_pivot = add_sample_statistics_per_fac_product(df_pivot)
    # make necessary cols numeric
    for i in df_pivot.columns:
        if i not in ['index', 'date', 'fac_name', 'fac_id', 'fac_type', 'id_dhis2'] + mfl_sel_cols:
            df_pivot[i] = pd.to_numeric(df_pivot[i], errors='coerce')

    # save
    if save:
        df_pivot.to_csv(os.path.join(path, fname), index=False)

    return df_pivot


def sep_factype_from_facname(df, fac_type_list, col='facility_name', return_factype=True,
                             return_facname=True, prefix='seperated_'):
    """
    Separate facility type from facility name
    """

    df_ = df.copy()
    df_[prefix + col] = df_[col].apply(lambda x: ' '.join(x.split(' ')[:-1]) if x.split(' ')[-1] in fac_type_list else x)

    df_[prefix + 'fac_type'] = df_[col].apply(lambda x: x.split(' ')[-1] if x.split(' ')[-1] in fac_type_list else np.nan)

    return df_


def add_hfpk_using_geo(df, lat_col='latitude', long_col='longitude',
                       score_thr=0.99, coor_col=None, filter_point_coor=True,
                       s3_mfl_link="data/AWS/raw/master_facility_update_9.csv"):
    """
    Add hf_pk from master facility list to  the given dataframe using geo location
    """

    df_geo = df.copy()
    df_mfl = pd.read_csv(s3_mfl_link)

    # only include point coordinates
    if filter_point_coor:
        mask = df_geo.featureType == 'POINT'
        df_geo = df_geo[mask].reset_index(drop=True)

    if coor_col != None:
        df_geo[lat_col] = [eval(i)[1] if i == i else np.nan for i in df_geo[coor_col].values]
        df_geo[long_col] = [eval(i)[0] if i == i else np.nan for i in df_geo[coor_col].values]

    # Create an indexer object
    indexer = recordlinkage.Index()
    indexer.full()

    # Create candidate pairs
    pairs = indexer.index(df_geo, df_mfl)

    # Create a comparison object
    compare = recordlinkage.Compare()

    # GEO
    compare.geo(left_on_lat=lat_col, left_on_lng=long_col, right_on_lat='lat', right_on_lng='long',
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


def deal_with_zeros_nans_inDHIS2(df, product_names,
                                 ops=['invalid_zeros', 'stock_sum_err', 'drop_stockout']):
    """
    Deal with zeros and nans in DHIS2 data with different strategies.
    selected rows will turn to nans for dispensed vals.

    invalid_zeros: All stock info (possibly with the exception of stock_ordered)
    are recorded as '0' for a given contraceptive product, month and service delivery site

    stock_sum_err: relations between stockout, Closing/Opening Balance, Received and
    dispensed doesn't add up

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
            sel_col = [i + j for j in sel_stock_abbr]
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

            # adjustment --> find rows that received + adjusted == dispensed + Closing balance
            BpC = B + C
            DpE = D + E
            mask2 = BpC == DpE
            mask = mask1 | mask2
            df_pivot.loc[~mask.values, i + '- Quantity Dispensed (D)'] = np.nan

    if 'drop_stockout' in ops:
        # remove (make nans) rows where stockout is positive

        #sel_stock_abbr = ['- Quantity Dispensed (D)',
        #                  '- Closing Balance (E)',
        #                  '- Days Out of Stock (F)',
        #                  '- Losses / Adjustments (C)',
        #                  '- Opening Balance (A)',
        #                  '- Quantity Received (B)']

        for i in product_names:
            sel_col = i + '- Days Out of Stock (F)'
            mask = df_pivot[sel_col] > 0
            df_pivot.loc[mask.values, i + '- Quantity Dispensed (D)'] = np.nan

    return df_pivot


def add_cols_from_mfl_on_hfpk(df, cols=['region', 'chiefdom', 'district'],
                              s3_mfl_link="data/AWS/raw/master_facility_update_9.csv"):
    """
    Merge cols from master health facilities to a given dataframe
    """
    df_master = pd.read_csv(s3_mfl_link)
    return df.merge(df_master[['hf_pk'] + cols], on='hf_pk')


def create_unpivot_dataframe(df_pivot, id_vars, key_word_valvars, var_name, val_name):
    val_vars = df_pivot.columns[df_pivot.columns.str.contains(key_word_valvars, regex=False)].tolist()
    df = df_pivot.melt(id_vars=id_vars, var_name=var_name, value_vars=val_vars, value_name=val_name)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    return df


def melt_multiplecols(df_pivot, id_vars, key_word_valvars, var_name, val_name, num_cols, product_names):

    if len(id_vars) != num_cols or len(key_word_valvars) != num_cols:
        raise ValueError("all input should be the same length")

    if len(var_name) != num_cols or len(val_name) != num_cols:
        raise ValueError("all input should be the same length")

    for k in range(num_cols):
        df = create_unpivot_dataframe(df_pivot, id_vars[k], key_word_valvars[k], var_name[k], val_name[k])

        for p in product_names:
            df.loc[df['product'].str.contains(p, regex=False), 'product'] = p

        if k == 0:
            df_final = df.copy()
        else:
            df_final = pd.merge(df_final, df, on=['fac_name', 'product', 'date'])

    return df_final


def extract_monthlyavg_fromDHIS2(df, m=3):
    """
    Extract monthly average values from DHIS2
    """

    sel_cols = ['date', 'fac_name'] + list(df.columns[df.columns.str.contains(str(m) + ' Months', regex=False)].values)
    df_monthavg = df[sel_cols].melt(['date', 'fac_name'],
                                    value_name='avg_' + str(m) + 'month_DHIS2').rename(columns={'variable': 'product'})

    df_monthavg['date'] = pd.to_datetime(df_monthavg['date'], errors='coerce')

    # change product name for a new column
    df_monthavg['product'] = df_monthavg['product'].str.replace('Average Monthly Consumption for ', '')
    df_monthavg['product'] = df_monthavg['product'].str.replace(' - ' + str(m) + ' Months', ' - Quantity Dispensed (D)')

    return df_monthavg


def load_clean_prepare_4ML(df_pivot, product_names, product_group_idx, thr_quantile=0.99):
    """
    Clean, preprocess and make the raw DHIS2 dat ready for ML
    """

    df_pivot = add_cols_from_mfl_on_hfpk(df_pivot, cols=['region', 'chiefdom', 'district'])

    # deal with zeros
    df_pivot = deal_with_zeros_nans_inDHIS2(df_pivot, product_names,
                                            ops=['invalid_zeros', 'stock_sum_err', 'drop_stockout'])
    # melt pivot table and create dataframe
    num_cols = 5
    id_vars = [['date', 'fac_name', 'fac_type', 'latitude', 'longitude', 'hf_pk', 'region', 'chiefdom', 'district']] + \
              [['date', 'fac_name']] * (num_cols - 1)
    key_word_valvars = ['Dispensed', '3 Months', '6 Months', 'num_nans', 'num_sample']
    var_name = ['product'] * num_cols
    val_name = ['quantity', 'avg_3months_DHIS2', 'avg_6months_DHIS2', 'num_nans', 'num_sample']

    df = melt_multiplecols(df_pivot, id_vars, key_word_valvars, var_name, val_name, num_cols, product_names)
    # drop NANs in train period
    df = df.sort_values(['date'])
    df = df.dropna(subset=['quantity'])
    df = df.fillna(0).reset_index(drop=True)

    # remove values above threshold
    df_nonzero = df[df.quantity != 0]
    df_thr = df_nonzero.groupby(['product', 'fac_type'])['quantity'].quantile(thr_quantile)

    for i in df_thr.index:
        mask = df['product'] == i[0]
        mask1 = df['fac_type'] == i[1]
        mask2 = df.quantity > df_thr[i]
        idx_to_remove = df[mask & mask1 & mask2].index
        df = df.drop(index=idx_to_remove).reset_index(drop=True)

    # add meds category
    #df['meds_category'] = np.nan
    #for k in product_group_idx.keys():
     #   sel_prods = [product_names[i] for i in product_group_idx[k]]
      #  mask = df['product'].isin(sel_prods)
       # df.loc[mask, 'meds_category'] = k

    return df

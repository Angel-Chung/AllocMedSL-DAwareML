import pandas as pd
import numpy as np
import pandas as pd
from datetime import *
import time
import csv
import os
from retina.feature_engineering import (
    add_rolling,
    add_prev_periods,
    split_dates,
    str_to_cat,
    add_deriv,
    create_label)

def CleanDhis2(data,product_names):
    fname = f'tmp/dhis2Clean.csv'
    if not os.path.isfile(fname):
        df_raw=data
        for i in range(0, len(product_names)):
            if i == 0:
                df = df_raw[df_raw['Data_name'].str.contains(
                    product_names[i], regex=False)]
                df['product']=product_names[i]
                print(product_names[i], len(df))
            else:
                tmp = df_raw[df_raw['Data_name'].str.contains(
                    product_names[i], regex=False)]
                tmp['product']=product_names[i]
                print(product_names[i], len(tmp))
                df = pd.concat([df, tmp])
        data=df
        data['date'] = pd.to_datetime(data['date'])
        data['status'] = np.where(data['Data_name'].str.contains("Opening"),"Opening Balance (A)",
                          np.where(data['Data_name'].str.contains("Received"), "Quantity Received (B)",
                                   np.where(data['Data_name'].str.contains("Dispensed"), "Quantity Dispensed (D)",
                                            np.where(data['Data_name'].str.contains("Losses"), "Losses / Adjustments (C)", 
                                                     np.where(data['Data_name'].str.contains("Closing"), "Closing Balance (E)", 
                                                              np.where(data['Data_name'].str.contains("Days"), "Days Out of Stock (F)", 
                                                                       np.where(data['Data_name'].str.contains("Stockout"), "Stockout","AMC")))))))

        AllEMshape = data.pivot_table(index=['Organisation unit', 'date', 'product'], columns='status',values="Value").reset_index()

        # Creating new columns based on conditions
        AllEMshape['zero'] = np.where((AllEMshape[('Opening Balance (A)')] == 0) & 
                                      (AllEMshape[('Quantity Received (B)')] == 0) & 
                                      (AllEMshape[('Closing Balance (E)')] == 0) & 
                                      (AllEMshape[('Quantity Dispensed (D)')] == 0) & 
                                      (AllEMshape[('Losses / Adjustments (C)')] == 0), 1, 0)

        AllEMshape['balance'] = np.where((AllEMshape[('Opening Balance (A)')] + 
                                           AllEMshape[('Quantity Received (B)')] - 
                                           AllEMshape[('Quantity Dispensed (D)')] + 
                                           AllEMshape[('Losses / Adjustments (C)')] == 
                                           AllEMshape[('Closing Balance (E)')]), 0, 1)

        AllEMshape['stockout'] = np.where((AllEMshape[('Days Out of Stock (F)')] > 0) | 
                                          (AllEMshape[('Stockout')] == 1), 1, 0)
        Allmiss = AllEMshape[(AllEMshape['zero'] == 0) & (AllEMshape['balance'] == 0) & (AllEMshape['stockout'] == 0)]
    
    else: 
        Allmiss=pd.read_csv(fname)
        
    return Allmiss, AllEMshape

def ProcessBFeature(data):
    facFeature = pd.read_csv("/Experiment/MFLhfpk_Dhis2AllQ4.csv")
    data['SOH'] = data['Closing Balance (E)']
    data['quantity'] = data['Quantity Dispensed (D)']
    data.rename(columns={'Organisation unit': 'organisationunit_id'}, inplace=True)
    data = data.merge(facFeature, how='left', on='organisationunit_id')
    Dhis2BFeature = data[['product','organisationunit_id', 'hf_pk', 'date', 'quantity', 'facility_type', 'lat', 'long', 'district', 'SOH']]
    Dhis2BFeature=Dhis2BFeature.dropna()
    return Dhis2BFeature

def create_features_essential_meds(df: pd.DataFrame,
                                   grouping_cols=['hf_pk', 'product'],
                                   date_column='date',
                                   lead_time=1,
                                   return_mapping=True) -> pd.DataFrame:
    """create features using forecasting library
    Args:
        df: PREPROCESSED dataframe
    Returns: df with ts features
    """
    df=df.copy()
    df= df.drop_duplicates()
    facility = df['hf_pk'].unique()
    products = df['product'].unique()

    # construct complete data set from 2019 March (they started recorded data better) until the latest date we have 
    start_date = pd.to_datetime("2019-03-01", format="%Y-%m-%d")
    end_date = pd.to_datetime("2023-05-01", format="%Y-%m-%d")
    dates=pd.date_range(start=start_date, end=end_date, freq='MS')
    data = pd.MultiIndex.from_product([facility, products, dates], names=['hf_pk', 'product', 'date'])
    df_complete = pd.DataFrame(index=data).reset_index()

    # extract and merge the facility features first 
    dfOrg= df[['hf_pk','facility_type','lat','long','district']]
    dfOrg= dfOrg.drop_duplicates()
    df_complete = pd.merge(df_complete, dfOrg, on=['hf_pk'], how='left')
    
    # extract and merge facility-product-date pair consumption 
    df['date']=df['date'].map(lambda x:datetime.strptime(x,"%Y-%m-%d"))
    df2=df.loc[:, ~df.columns.isin(['facility_type','lat','long','district'])]
    df2 = df2.drop_duplicates(subset=['hf_pk','date','product'])
    df_complete2 = pd.merge(df_complete, df2, on=['hf_pk','date','product'], how='left')


    # start creating the features (mostly use their functions)
    df = df_complete2
    quantity_column = 'quantity'
    df = add_rolling(df, date_column, grouping_cols,
                         quantity_column, [2, 3, 4, 5, 6], rolling_stat='mean')
    df = add_prev_periods(df, date_column, grouping_cols,
                          quantity_column, 6)
    df = add_rolling(df, date_column, grouping_cols,
                         quantity_column, [3, 6], rolling_stat='std')
    df = add_rolling(df, date_column, grouping_cols,
                         quantity_column, output_name='total_sample', 
                         rolling_stat='count')
    df = add_rolling(df, date_column, grouping_cols,
                         quantity_column,  [3, 6, 12], 
                         rolling_stat='count')
    df = split_dates(df, date_column)

    df = add_deriv(df, date_column, grouping_cols,
                   quantity_column, 3)
    df = add_rolling(df, date_column, ['product'],
                     quantity_column, [2, 3, 4, 5, 6, 10], output_name='avg_per_product')

    if return_mapping:
        df, mapping = str_to_cat(df, return_mapping=True)
    else:
        df = str_to_cat(df)

    # create target 
    df = create_label(df, date_column, grouping_cols,target_column=quantity_column,lead_time=lead_time, mode='test')
    df= df.drop_duplicates()
  
    #adding to our date the total lead time!
    df['date'] = df.date+ pd.DateOffset(months=lead_time)

    if return_mapping:
        return df, mapping
    return df


import pandas as pd
import numpy as np

# create historical variance file for fac-product pair
def getHistVar(data, date): # input df4ML data 
    var = data[(data['date'] > "2019-02-01") & (data['date'] < date[0])]
    var = var[['product', 'date', 'hf_pk', 'target']]
    var2 = var.groupby(['product', 'hf_pk']).agg({'target':lambda x: x.std(skipna=True)}).reset_index().rename(columns={'target':'standardD'})

    # Replace NaN values with 0
    var2['standardD'] = var2['standardD'].fillna(0)
    return var2

# create budget based on previous consumption*3 
def getBudget(data,date): # input df4ML data
    data=data[data['date']==date[0]].copy()
    df = data.groupby('product').agg({'target':lambda x: x.sum(skipna=True)}).rename(columns={'target':'stock'}).reset_index()
    df['stock']=df['stock']*3
    df = df.drop_duplicates()

    return df

def GETdf4ML():
    fname = f'RawCleanDhis2OG_Q2.csv'
    if not os.path.isfile(fname):
        data = pd.read_csv("/Experiment/dfRaw_2023Q2.csv") 
        dhis2, dhis2OG = CleanDhis2(data,product_names)
        dhis2.to_csv(f'RawCleanDhis2_Q2.csv')
        dhis2OG.to_csv(f'RawCleanDhis2OG_Q2.csv')
    
    dhis2 = pd.read_csv(fname)
    df = dhis2
    BeforeFeature = ProcessBFeature(df) 
    BeforeFeature.to_csv(f'Dhis2BFfeature2023Q2.csv')
    df4ML, mapping=create_features_essential_meds(BeforeFeature)
    return df4ML, mapping

    
def main():
    fname = f'df4ML.csv' 
    if not os.path.isfile(fname):
        df4ML, mapping=GETdf4ML()
        df4ML.to_csv(f'df4ML.csv')
        with open('mapping.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for key, value in mapping.items():
                writer.writerow([key, value])


if __name__ == "__main__":
    main()

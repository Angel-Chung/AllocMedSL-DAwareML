from conf.config import *
# add the path for forecasting libraray <-- this will be removed after pip install
import sys; sys.path.append(FORECAST_LIB_DIR)
from retina.loading import *
import pandas as pd
import numpy as np
import os

#sys.path.append('..')

def fetch_preprocessedDHIS2_EssMeds(preprocessed_file_name, file_version=2):
    """load DHIS2 data, clean and prepare it for ML"""
    file_path = os.path.join(LOCAL_PATH, preprocessed_file_name + '_v'+str(file_version)+'.csv')
    if os.path.exists(file_path):
        df_pivot = pd.read_csv(file_path)
    else:
        to_drop = ['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1',
                   'Unnamed: 0.2']  # , 'Unnamed: 0.1.1.1']
        df_pivot = load_fromS3_convert_clean_save(
            LOCAL_PATH, S3_LINK + DHIS2_PATH, S3_LINK +
            GEOSPATIAL_PATH, product_names, fac_type_list,
            save=True, fname=preprocessed_file_name+'_v'+str(file_version)+'.csv', to_drop=to_drop)
    df = load_clean_prepare_4ML(df_pivot, product_names, product_group_idx,
                                thr_quantile=0.95)

    return df

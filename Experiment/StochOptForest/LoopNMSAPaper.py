#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tree import *
from nv_tree_utilities import *
from joblib import dump

import mkl
mkl.set_num_threads(1)

seed = 0
np.random.seed(seed)

def getBudget(data,date): # input df4ML data
    data=data[data['date']==date[0]].copy()
    df = data.groupby('product').agg({'target':lambda x: x.sum(skipna=True)}).rename(columns={'target':'stock'}).reset_index()
    df['stock']=df['stock']*3
    df = df.drop_duplicates()
    return df
p_list = [38] # covariates
runs = 1
n_jobs = 1
n_trees = 500
N_list = [1,1,1,1,1,1,1,1]

b_list = np.array([1.]) # backorder cost
h_list = np.array([0.])

L = len(h_list)
honesty = False; 
verbose = False; oracle = True;
bootstrap = True; 

risk_all = {}
results_eval_all = {}

direct = '/Experiment/StochOptForest'
date = '2024'
output = direct + date + "nv_n.txt"
Tdate=['2023-01-01'] #you can adjust for other test date: Our baseline is taking the average of results from 2022-12-01 to 2023-02-01

df=pd.read_csv("/Experiment/df4ml.csv")
drop_cols = ['Unnamed: 0']
df = df.drop(drop_cols, axis = 1)
facility=[246,258,299,613,697,721,10116,10286,10324,30387,638] # remove outlier facility
df = df[~df["hf_pk"].isin(facility)]
df = df[df['date'] > '2019-03-01'].copy()
df = df[df['date'] < '2023-02-01']
df=df.dropna()

date_to_int_mapping = {date: i for i, date in enumerate(df['date'].unique(), 1)}
df['date'] = df['date'].map(date_to_int_mapping)
TdateID=[df['date'].max()]
Test=df[df['date']==TdateID[0]].copy()

Train=df[df['date']<TdateID[0]].copy()
with open(output, 'w') as f:
    print("start", file = f)
budget=['Real25','Real75','RealMed']
for bgt in budget:
    if bgt=="AvgHist3":
        budget2=getBudget(data,date)
    elif bgt=="AvgHist1":
        budget=getBudget(data,date)
        budget2=budget['stock']/3
    else:
        budget=pd.read_csv(f'/Experiment/budget2023Q2Script_processed.csv')
        budget2=budget[budget['budgetType']==bgt].copy()
        dual=pd.read_csv(f'/Experiment/tmp/dual_{str(Tdate[0])}.csv')
    for i,N in zip(sorted(df['product'].unique()),N_list):
        print(i)
        budget3=budget2[(budget2['product']==i)].copy()
        print(budget3)
        C=list(budget3['stock'])
        C=C[0]

        LBDA=dual[(dual['product']==i)].copy()
        lbda=list(LBDA['dual'])

        risk_all[str(N)] = {}
        results_eval_all[str(N)] = {}

        subsample_ratio = 1;
        max_depth=100;
        min_leaf_size=10;
        balancedness_tol = 0.2;
        n_proposals = N;
        mtry = 38; 
            
        np.random.seed(seed)
        X_list=Train[Train['product']==i].copy()
        X_list=X_list.drop(['target','product'],axis=1)
        X_list  = [np.array(X_list) for run in range(runs)]

        Y_list = Train[Train['product']==i].copy()
        Ny_train=Y_list.shape[0]

        Y_list = Y_list[['target']]
        Y_list = [np.array(Y_list) for run in range(runs)]

        X_val=Test[Test['product']==i].copy()
        Nx_test=X_val.shape[0]
        Ny_test=X_val.shape[0]
        X_val=X_val.drop(['target','product'],axis=1)
        X_val  = [np.array(X_val) for run in range(runs)]
                                       
        Y_est=Test[Test['product']==i].copy()
        Y_est=Y_est[['target']]
        Y_val  = [np.array(Y_est) for run in range(runs)]

        time1 = time.time()
        results_fit = Parallel(n_jobs=n_jobs, verbose = 3)(delayed(compare_forest_one_run)(X_list[run], Y_list[run], X_val[run], Y_val[run],h_list = h_list, b_list = b_list, C = C, n_trees = n_trees, honesty= honesty, mtry = mtry, subsample_ratio = subsample_ratio, oracle = oracle, min_leaf_size = min_leaf_size, verbose = verbose, max_depth = max_depth, n_proposals = n_proposals, balancedness_tol = balancedness_tol, bootstrap = bootstrap, seed = seed,lbda=lbda) for run in range(runs))
        time2 = time.time()
        print("fitting time",time2 - time1)

        results_eval = Parallel(n_jobs=n_jobs, verbose = 3)(delayed(evaluate_one_run)(results_fit[run], X_list[run], Y_list[run], X_val[run], Y_val[run],Nx_test, Ny_train, Ny_test,h_list =h_list, b_list = b_list, C = C, verbose = verbose, seed =seed,lbda=lbda) for run in range(runs))
        print(np.shape(results_eval[0][0]['rf_approx_sol']))
        print(results_eval[0][0]['rf_approx_sol'])

        pickle.dump(results_eval, open(direct + Tdate[0] +str(i) +str(bgt)+"NMSAPaper.pkl", "wb")) # You need to process and combine the results
        with open(direct + Tdate[0] +str(i) +str(bgt)+"NMSAPaper.pkl", 'rb') as file:
            data = pickle.load(file)
        rf_approx_sol_array = data[0][0]['rf_approx_sol']
        rf_approx_sol_flattened = rf_approx_sol_array.flatten()
        test=Test[Test['product']==i].copy()
        test['demand']=rf_approx_sol_flattened
        test.to_csv(f'/Experiment/StochOptForest/tmp/2023-01-01_StochForest_{str(i)}_{str(bgt)}Q2.csv')
    
        

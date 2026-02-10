from datetime import *
import numpy as np
import pandas as pd
import logging
import argparse
import os
from sklearn.ensemble import RandomForestRegressor
import warnings
import cvxpy as cp
import time
import scipy
import scs
import scipy.stats as st
import pickle

# save allocations for given demand prediction approach
def get_allocation(df, n_estimators, allocation_max, date, product, optimize_fn):
    df = df[(df['product'] == product) & (df['date'] == date)].copy()
    allocation_max=df['stock'].unique()
    allocation = optimize_fn(df, n_estimators, allocation_max,product)
    df['allocation'] = allocation
    return df
    
    
# optimize allocations using linear programming
def optimize_lp(demand, allocation_max, product):
    n_facilities = len(demand)

    if np.shape(demand)[0]!=0:
        opt_type = 'none'
        demand_multiplier = 1.0

        # linear program
        allocation = cp.Variable(shape=(n_facilities), name="allocation")
        loss = cp.Variable(shape=(n_facilities), name="loss")
        constraints = [allocation >= 0, cp.sum(allocation) <= allocation_max, loss >= 0, loss >= demand * demand_multiplier - allocation]
      
        objective = cp.Minimize(cp.sum(loss))
            
        # solution
        prob = cp.Problem(objective, constraints)
        prob.solve(cp.GUROBI)
    else:
        print("no trees")
        return None
        
    return allocation.value

# construct demand samples according to LP with tree demand distribution
def optimize_fn_ours(df, n_estimators, allocation_max, product):
    p=np.array(df['demand'])
    p = np.nan_to_num(p)
    rng=np.random.default_rng(10)
    demandN=[]
    var=df['standardD']
    var=np.array(var)
    for i in range(len(p)):
        if var[i]>0:
            d=st.norm.rvs(p[i],var[i]/10,size=n_estimators,random_state=rng)
            demandN.append(d)
        else:
            d=st.norm.rvs(p[i],0,size=n_estimators,random_state=rng)
            demandN.append(d)

    demandN=np.array(demandN)
    return optimize_lp(p, allocation_max, product)

# optimize demand using proportional allocation
def optimize_fn_prop(df, n_estimators, allocation_max):
    demand = np.array([df[f'demand{i}'] for i in range(n_estimators)]).T
    demand = np.sum(demand, axis=1)
    return allocation_max * demand / np.sum(demand)

# optimize demand using proportional allocation
def optimize_fn_oracle(df, n_estimators, allocation_max):
    demand = np.array([df['target']]).T
    return optimize_lp(demand, allocation_max)

# evaluate unmet demand
def evaluate(df):
    return np.sum(np.maximum(df['target'] - df['allocation'], 0.0))

def get_stockouts(allocation, target):
    alloc = allocation[allocation['target'] > allocation['allocation']]
    return set(alloc['fac_id'])

def get_allocation_all(df, n_estimators, allocation_max, optimize_fn):
    df_allocation = []
    for date in sorted(df['date'].unique()):
        for product in range(8):
            df_cur = get_allocation(df, n_estimators, allocation_max, date, product, optimize_fn)
            print(date, product, evaluate(df_cur))
            df_allocation.append(df_cur)
    return pd.concat(df_allocation, ignore_index = True)
    


def parser():

    # set-up parsers/subparsers
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Get predictions
    predictions = subparsers.add_parser('get-allocations')
    predictions.add_argument('--d', '--date', type=str,nargs='+',
                             help='Months you want for predictions, in YYYY-MM-01 form.')
    predictions.add_argument('--agg', '--aggregation', type=str,
                             help='aggregation level, e.g. district')
    predictions.add_argument('--fu', '--force-update', action='store_true',
                             help='Force update. If added, will overwrite existing files.')
    predictions.add_argument('--l', '--leadtime', type=int,
                             help='leadtime for forecasting. 1 means next month, 2 means two month ahead, etc.')
    return parser.parse_args()


def main(args):
    time1=time.time()
    allocation_max = 0 # adjust in get_allocation function
    n_estimators = 500
    dates=list(args.date)
    date_column='date'
    bgt=['Real25','Real75','RealMed']
    target_col='target'
    lead_time=1
    approaches = [('ours', optimize_fn_ours)]
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    
    # Step 1: Get predictions
    df = pd.read_csv("/Experiment/df4ml.csv")
    facility=[246,258,299,613,697,721,10116,10286,10324,30387,638] # remove outlier facility
    df = df[~df["hf_pk"].isin(facility)]
    df = df.dropna()
    
    stoch = pd.read_csv(f'/Experiment/StochOptForest/tmp/AllPdQ2_StochForestAllegro.csv') #Obtained from running LoopNMSAPaper.py
    for dateRun in dates: 
        test=df[df['date']==dateRun]
        var = getHistVar(df,dates)
        test = test.merge(var, on=['product','hf_pk'], how='left')
        test["standardD"] = test["standardD"].fillna(0)
        stoch2=stoch[stoch['date']==dateRun].copy()

        
        for budgt in bgt: 
            if budgt=="AvgHist3":
                #budget=pd.read_csv(f'tmp/budgetConsump3_2022-12-01.csv')
            elif budgt=="AvgHist1":
                #budget=pd.read_csv(f'tmp/budgetConsump3_2022-12-01.csv')
                budget['stock']=budget['stock']/3
            else:
                budget=pd.read_csv(f'/Experiment/budget2023Q2Script_processed.csv')
                budget=budget[budget['budgetType']==budgt].copy()

            stoch3=stoch2[stoch2['budgetType']==budgt].copy()
            stoch3=stoch3[['hf_pk','product','demand']]
            test2 = test.merge(stoch3, on=['product','hf_pk'], how='left')
            test3=test2.merge(budget, on=['product'], how='left')
            test3['demand']=test3['demand'].fillna(method='ffill')

            for name, optimize_fn in approaches:
                df_allocation = get_allocation_all(test3, n_estimators, allocation_max, optimize_fn)
                df_allocation.to_csv(f'results/allocationQ2Pd_StochForest_{str(dateRun)}_{str(budgt)}.csv')
                print(name, evaluate(df_allocation))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', nargs='+', type=str, choices=['2023-01-01','2023-02-01','2022-12-01'],default='2023-01-01')
    args = parser.parse_args()
    main(args)

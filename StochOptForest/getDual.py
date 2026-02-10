from src.model import get_training_data, train_model
from Preprocess import *
from datetime import *
import numpy as np
import pandas as pd
import logging
import argparse
import os
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import warnings
import cvxpy as cp
import time
import scipy
import scipy.stats as st

def get_predictions(data, dates, n_estimators, date_column, target_col, lead_time,df_weight = None, aggregation=None):
    print('constructing dataset')
    df = data
    drop_cols = ['SOH','Avg6mth','Avg3mth','upper','exclude']
    df = df.drop(drop_cols, axis = 1)
    facility=[246,258,299,613,697,721,10116,10286,10324,30387,638] # remove outlier facility
    df = df[~df["hf_pk"].isin(facility)]
    df['date']=df['date'].map(lambda x:datetime.strptime(x,"%Y-%m-%d"))
    print(np.shape(df))
    df = df[df['date'] > '2019-02-01']
    for i in range(len(dates)):
        Tdate = dates[i]
        print('date',Tdate)
        train = df[(df['date'] + pd.offsets.DateOffset(months=(lead_time-1))) < Tdate]
        if train.isna().sum().sum() > 0:
            train = train.dropna()
            warnings.warn('data include Nan values. Dropped from train sets!')
        print(f"Training min: {train['date'].min()}")
        print(f"Training max: {train['date'].max()}")
        val = df[df['date'] == Tdate]
        val = val.dropna()
        print(f"Test min: {val['date'].min()}")
        print(f"Test max: {val['date'].max()}")
        print(f"Total sample test: {len(val)}")
    if not df_weight is None:
        df_weight['date']=df_weight['date'].astype('datetime64[ns]')
        train = pd.merge(train, df_weight, on=['hf_pk', 'date', 'product'], how = 'left')
        train['weight'] = train['weight'].fillna(0)
    rfr = train_model(train, n_estimators)
    xs_train, _, _ = get_training_data(train)
    xs_test, _, _ = get_training_data(val)
    for i, tree in enumerate(rfr.estimators_):
        train[f'demand{i}'] = np.maximum(tree.predict(xs_train), 0)
        val[f'demand{i}'] = np.maximum(tree.predict(xs_test), 0)
    return train, val



def get_allocation(df, n_estimators, allocation_max, date, product, optimize_fn):
    df = df[(df['product'] == product) & (df['date'] == date)].copy()
    allocation_max=df['stock'].unique()
    print(allocation_max)
    allocation, dual = optimize_fn(df, n_estimators, allocation_max,product)
    print(dual)
    df['allocation'] = allocation
    df['laguagian'] = dual[0]
    return df
    
    
# optimize allocations using linear programming
def optimize_lp(demand, allocation_max, product):
    n_facilities, n_samples = demand.shape

    if np.shape(demand)[0]!=0:
        opt_type = 'none'
        demand_multiplier = 1.0

        # linear program
        allocation = cp.Variable(shape=(n_facilities), name="allocation")
        loss = cp.Variable(shape=(n_facilities, n_samples), name="loss")
        constraints = [cp.sum(allocation) <= allocation_max, allocation >= 0,loss >= 0]
       
        for i in range(n_samples):
            constraints += [loss[:,i] >= demand[:,i] * demand_multiplier - allocation]
        
        if opt_type == 'cvar_samples':
            loss_cvar = cp.Variable(shape=(n_samples), name="loss_cvar")
            z_cvar = cp.Variable(name="z_cvar")
            for i in range(n_samples):
                constraints += [loss_cvar[i] >= cp.sum(loss[:,i]) - z_cvar]
                constraints += [loss_cvar[i] >= 0]
            objective = cp.Minimize(z_cvar + 1.1 * cp.sum(loss_cvar) / n_samples)
        elif opt_type == 'cvar_facilities':
            loss_cvar = cp.Variable(shape=(n_facilities), name="loss_cvar")
            z_cvar = cp.Variable(name="z_cvar")
            for i in range(n_facilities):
                constraints += [loss_cvar[i] >= cp.sum(loss[i,:]) - z_cvar]
                constraints += [loss_cvar[i] >= 0]
            objective = cp.Minimize(z_cvar + 1.1 * cp.sum(loss_cvar) / n_facilities)
        elif opt_type == 'quad':
            objective = cp.Minimize(cp.sum_squares(loss))
        else:
            objective = cp.Minimize(cp.sum(loss))
            
        objective = cp.Minimize(cp.sum(loss))
        # solution
        prob = cp.Problem(objective, constraints)
        if opt_type == 'quad':
            try:
                prob.solve()
            except:
                print('error with quadratic, trying linear')
                objective = cp.Minimize(cp.sum(loss))
                prob = cp.Problem(objective, constraints)
                prob.solve(cp.SCIPY, scipy_options={"method": "highs"})
        else:
            prob.solve(cp.GUROBI)
    else:
        print("no trees")
        return None
        
    return allocation.value, constraints[0].dual_value


def optimize_fn_ours(df, n_estimators, allocation_max, product):
    demand = np.array([df[f'demand{i}'] for i in range(n_estimators)]).T
    p=np.apply_along_axis(st.norm.fit, axis=1, arr=demand)
    rng=np.random.default_rng(10)
    demandN=[]

    var=df['standardD']
    var=np.array(var)
    for i in range(len(p)):
        if var[i]>80:
            d=st.norm.rvs(p[i,0],var[i],size= n_estimators,random_state=rng)
            demandN.append(d)
        else:
            d=st.norm.rvs(p[i,0],80,size= n_estimators,random_state=rng)
            demandN.append(d)

    demand_mean = np.mean(demand, axis=1)
    demand_mean = np.array([demand_mean for _ in range(n_estimators)]).T
    demand = demand_mean + (demand - demand_mean) * 1.0
    demandN=np.array(demandN)
    return optimize_lp(demandN, allocation_max, product)

def get_allocation_all(df, n_estimators, allocation_max, optimize_fn):
    df_allocation = []
    df = df[df['date'] != '2019-02-01']
    df=df[pd.notnull(df['date'])]
    for date in sorted(df['date'].unique()):
        print(date)
        for product in range(8):
            df_cur = get_allocation(df, n_estimators, allocation_max, date, product, optimize_fn)
            print(date, product, evaluate(df_cur))
            df_allocation.append(df_cur)
    return pd.concat(df_allocation, ignore_index = True)


def evaluate(df):
    return np.sum(np.maximum(df['target'] - df['allocation'], 0.0))


def main():
    allocation_max = 0 # adjust in get_allocation function
    n_estimators = 500
    dates=['2022-11-01']
    date_column='date'
    target_col='target'
    lead_time=1
    approaches = [('ours', optimize_fn_ours)]
    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    # import data
    fname = f'/Experiment/df4ml.csv' 
    df4ML=pd.read_csv(fname)

    train, test = get_predictions(df4ML, dates, n_estimators, date_column, target_col,lead_time)
    var = getHistVar(df4ML,dates)
    test = test.merge(var, on=['product','hf_pk'], how='left')
    test["standardD"] = test["standardD"].fillna(0)

    budget=getBudget(df4ML,dates)
    test=test.merge(budget, on=['product'], how='left')

    for name, optimize_fn in approaches:
        df_allocation = get_allocation_all(test, n_estimators, allocation_max, optimize_fn)
        print(df_allocation)
        df_allocation =df_allocation[['product','dual']]
        df_allocation =df_allocation.drop_duplicates()
        df_allocation.to_csv(f'tmp/dual_{str(dates[0])}.csv')

if __name__ == "__main__":
    main()

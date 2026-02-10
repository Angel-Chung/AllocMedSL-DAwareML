from src.model import get_training_data, train_model
from retina.metrics import *
from Preprocess import *
from datetime import *
import numpy as np
import pandas as pd
import logging
import argparse
import os
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RandomizedSearchCV
import warnings
import cvxpy as cp
import time
import scipy
import scs
import scipy.stats as st

def get_predictions(data,date, n_estimators, date_column, target_col, lead_time,df_weight = None, aggregation=None):
    print('constructing dataset')

    df = data.copy()
    facility=[246,258,299,613,697,721,10116,10286,10324,30387,638] #remove outlier facility
    df = df[~df["hf_pk"].isin(facility)]
    date=datetime.strptime(date, "%Y-%m-%d")
    df['date']=df['date'].map(lambda x:datetime.strptime(x,"%Y-%m-%d"))
    train = df[(df['date'] + pd.offsets.DateOffset(months=(lead_time-1))) < date]
    if train.isna().sum().sum() > 0:
        train = train.dropna()
        warnings.warn('data include Nan values. Dropped from train sets!')
    print(f"Training min: {train['date'].min()}")
    print(f"Training max: {train['date'].max()}")
    val = df[df['date'] == date]
    val = val.dropna()
    features = train.columns.tolist()
    val=val[features]
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


# save allocations for given demand prediction approach
def get_allocation(df, n_estimators, allocation_max, date, product, optimize_fn):
    df = df[(df['product'] == product) & (df['date'] == date)].copy()
    allocation_max=(df['stock'].unique())
    allocation = optimize_fn(df, n_estimators, allocation_max,product)
    df['allocation'] = allocation
    return df
    
    
# optimize allocations using linear programming
def optimize_lp(demand, allocation_max, product,scale):
    n_facilities, n_samples = demand.shape

    if np.shape(demand)[0]!=0 and allocation_max>0:
        demand=demand*scale
        opt_type = 'none'
        demand_multiplier = 1.0
        # linear program
        allocation = cp.Variable(shape=(n_facilities), name="allocation")
        loss = cp.Variable(shape=(n_facilities, n_samples), name="loss")
        constraints = [allocation >= 0, cp.sum(allocation) <= allocation_max, loss >= 0]
    
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
        
    return allocation.value

# construct demand samples according to LP with tree demand distribution
def optimize_fn_ours(df, n_estimators, allocation_max, product):
    df['RFprediction'] = df[[f'demand{i}' for i in range(n_estimators)]].mean(axis=1)
    scale=allocation_max/df['RFprediction'].sum()
    demand = np.array([df[f'demand{i}'] for i in range(n_estimators)]).T
    print(demand.shape)
    if np.shape(demand)[0]!=0 & np.shape(allocation_max)[0]!=0:
        p=np.apply_along_axis(st.norm.fit, axis=1, arr=demand)
        rng=np.random.default_rng(10)
        demandN=[]
        var=df['standardD']
        var=np.array(var)
        for i in range(len(p)):
            if var[i]>50:
                d=st.norm.rvs(p[i,0],var[i],size= n_estimators,random_state=rng)
                demandN.append(d)
            else:
                d=st.norm.rvs(p[i,0],50,size= n_estimators,random_state=rng)
                demandN.append(d)
    
        demand_mean = np.mean(demand, axis=1)
        demand_mean = np.array([demand_mean for _ in range(n_estimators)]).T
        demand = demand_mean + (demand - demand_mean) * 1.0
        demandN=np.array(demandN)
    else:
        demandN=demand
    return optimize_lp(demandN, allocation_max, product,scale)

# optimize demand using dhis2 3 month rolling average
def optimize_fn_3mthAvg(df, n_estimators, allocation_max):
    demand = np.array([df['avg_3months_DHIS2']]).T
    return optimize_lp(demand, allocation_max)

# evaluate unmet demand
def evaluate(df):
    return np.sum(np.maximum(df['target'] - df['allocation'], 0.0))

def get_stockouts(allocation, target):
    alloc = allocation[allocation['target'] > allocation['allocation']]
    return set(alloc['fac_id'])

def get_allocation_all(df, n_estimators, allocation_max, optimize_fn):
    df_allocation = []
    df = df[df['date'] != '2019-02-01']
    df=df[pd.notnull(df['date'])]
    # For computation and multi-task learning, we select subset of products to run respectively
    pdlist =[0,27,31,32,38,45,58,59,60,62,83,48,61,15,82,3,21,13,75,76,77,20 ,4,66,44,30,68,1,7,56,29,36,37,18,46,50,63,79]
    for date in sorted(df['date'].unique()):
        print(date)
        for product in sorted(pdlist):
            df_cur = get_allocation(df, n_estimators, allocation_max, date, product, optimize_fn)
            print(date, product, evaluate(df_cur))
            df_allocation.append(df_cur)
    return pd.concat(df_allocation, ignore_index = True)
    

def main(args):
    allocation_max = 0 # adjust in get_allocation function
    n_estimators = 500
    dates=list(args.date)
    bgt=args.budgetType
    date_column='date'
    target_col='target'
    lead_time=1
    approaches = [('ours', optimize_fn_ours)] 
    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    # import data
    fname = f'/Experiment/df4ml.csv'
    df4ML=pd.read_csv(fname)
    
    for dateRun in dates:

        # Step 1: Get predictions
        train, test = get_predictions(df4ML, dateRun, n_estimators, date_column, target_col,lead_time)
        
        var = getHistVar(df4ML,dateRun)
        train = train.merge(var, on=['product','hf_pk'], how='left')
        test = test.merge(var, on=['product','hf_pk'], how='left')
        train["standardD"] = train["standardD"].fillna(0)
        test["standardD"] = test["standardD"].fillna(0)
        
        if bgt[0]=="AvgHist3":
            budget=getBudget(df4ML,dates)
            print(budget)
        elif bgt[0]=="AvgHist1":
            budget=getBudget(df4ML,dates)
            budget=budget['stock']/3
        else:
            budget=pd.read_csv(f'/Experiment/budget2023Q2Script_processed.csv')
            budget=budget[budget['budgetType']==bgt[0]].copy()
        train=train.merge(budget, on=['product'], how='left')
        test=test.merge(budget, on=['product'], how='left')

        # Step 3: Optimization
        if not os.path.exists('result'):
            os.mkdir('result')
        for name, optimize_fn in approaches:
            df_allocation = get_allocation_all(test, n_estimators, allocation_max, optimize_fn)
            df_allocation.to_csv(f'result/allocation_{str(dateRun)}_{str(bgt)}_Q2OGNMSAPaper.csv')
            print(name, evaluate(df_allocation))
 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', nargs='+', type=str, choices=['2023-01-01','2023-02-01','2022-12-01'],default='2023-01-01')
    parser.add_argument('--budgetType', nargs='+', type=str, choices=['AvgHist3','RealAvg','Real25','AvgHist1','Real75','RealMed'],default='Real25')
    args = parser.parse_args()
    main(args)

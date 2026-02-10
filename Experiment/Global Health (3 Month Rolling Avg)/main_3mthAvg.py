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
import scipy.stats as st


# save allocations for given demand prediction approach
def get_allocation(df, n_estimators, allocation_max, date, product, optimize_fn):
    df = df[(df['product'] == product) & (df['date'] == date)].copy()
    allocation_max=(df['stock'].unique())
    allocation = optimize_fn(df, n_estimators, allocation_max,product)
    df['allocation'] = allocation
    return df
    
    
# optimize allocations using linear programming
def optimize_lp(demand, allocation_max, product):
    n_facilities, n_samples = demand.shape
    if np.shape(demand)[0]!=0 and allocation_max>0:
        opt_type = 'none'
        demand_multiplier = 1.0
        # linear program
        allocation = cp.Variable(shape=(n_facilities), name="allocation")
        loss = cp.Variable(shape=(n_facilities, n_samples), name="loss")
        #fulfill = cp.Variable(shape=(n_facilities, n_samples), name="fulfill")
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
    demand = np.array([df[f'demand{i}'] for i in range(n_estimators)]).T
    if np.shape(demand)[0]!=0:
        p=np.apply_along_axis(st.norm.fit, axis=1, arr=demand)
        rng=np.random.default_rng(10)
        demandN=[]
        var=df['standardD']
        var=np.array(var)
        for i in range(len(p)):
            if var[i]>90:
                d=st.norm.rvs(p[i,0],var[i],size= n_estimators,random_state=rng)
                demandN.append(d)
            else:
                d=st.norm.rvs(p[i,0],90,size= n_estimators,random_state=rng)
                demandN.append(d)
    
        demand_mean = np.mean(demand, axis=1)
        demand_mean = np.array([demand_mean for _ in range(n_estimators)]).T
        demand = demand_mean + (demand - demand_mean) * 1.0
        demandN=np.array(demandN)
    else:
        demandN=demand
    return optimize_lp(demandN, allocation_max, product)

# optimize demand using dhis2 3 month rolling average
def optimize_fn_3mthAvg(df, n_estimators, allocation_max, product):
    demand = np.array([df['quantity_mean_3']]).T
    return optimize_lp(demand, allocation_max,product)

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
    for date in sorted(df['date'].unique()):
        print(date)
        for product in sorted(df['product'].unique()):
            df_cur = get_allocation(df, n_estimators, allocation_max, date, product, optimize_fn)
            print(date, product, evaluate(df_cur))
            df_allocation.append(df_cur)
    return pd.concat(df_allocation, ignore_index = True)
    

def main(args):
    allocation_max = 0 # adjust in get_allocation function
    n_estimators = 500
    dates=list(args.date)
    date_column='date'
    bgt=['AvgHist3']
    target_col='target'
    lead_time=1
    approaches = [('3mthAvg', optimize_fn_3mthAvg)]
    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    # import data
    df = pd.read_csv("/Experiment/df4ml.csv")
    df = df.dropna()

    for dateRun in dates: 
        test=df[df['date']==dateRun]
        for budgt in bgt: 
            print(budgt)
            if budgt=="AvgHist3":
                #budget=pd.read_csv(f'tmp/budgetConsump3_2022-12-01.csv')
            elif budgt=="AvgHist1":
                #budget=pd.read_csv(f'tmp/budgetConsump3_2022-12-01.csv')
                budget['stock']=budget['stock']/3
            else:
                budget=pd.read_csv(f'/Experiment/budget2023Q2Script_processed.csv')
                budget=budget[budget['budgetType']==budgt].copy()
            test3=test.merge(budget, on=['product'], how='left')


            # Step 3: Optimization
            for name, optimize_fn in approaches:
                df_allocation = get_allocation_all(test3, n_estimators, allocation_max, optimize_fn)
                df_allocation.to_csv(f'/Experiment/Global Health (3 Month Rolling Avg)/results/allocationQ2pd_3mthAvg_{str(dateRun)}_{str(budgt)}.csv')
                print(name, evaluate(df_allocation))
       
 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', nargs='+', type=str, choices=['2023-01-01','2023-02-01','2022-12-01'],default='2023-01-01')
    args = parser.parse_args()
    main(args)

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
import time
import scipy
import scs
import scipy.stats as st
import cvxpy as cp
import gurobipy


def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [
        st.alpha,st.beta,st.betaprime,st.chi,st.chi2,st.cosine,st.dgamma,st.dweibull,st.erlang,
        st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.genlogistic,st.genpareto,st.gennorm,
        st.genexpon,st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.laplace,st.logistic,st.loggamma,st.loglaplace,
        st.lognorm,st.lomax,st.maxwell,st.nakagami,st.norm,st.pareto,st.pearson3,st.powerlaw,
        st.powerlognorm,st.reciprocal,st.triang,st.tukeylambda,st.uniform,st.weibull_min,st.weibull_max
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf
    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)


def get_allocation(df, n_estimators, allocation_max,product, optimize_fn,dates):
    df = df[(df['product'] == product)].copy()
    allocation_max=df['stock'].unique()
    test=df[df['date']==dates]
    if len(test)!=0 and allocation_max>0:
        train=df[(df['date']<dates)]
        allocation = optimize_fn(test,train, n_estimators, allocation_max,product)
        test['allocation'] = allocation
    else:
        test['allocation'] = None
    return test
    
    
# optimize allocations using linear programming
def optimize_lp(demand, allocation_max,product):
    # variables
    n_facilities, n_samples = demand.shape

    if np.shape(demand)[0]!=0 and allocation_max!=0:
        opt_type = 'none'
        demand_multiplier = 1.0

        # linear program
        allocation = cp.Variable(shape=(n_facilities), name="allocation")
        loss = cp.Variable(shape=(n_facilities, n_samples), name="loss")
       
        constraints = [allocation >= 0, cp.sum(allocation) <= allocation_max, loss >= 0]
    
        for i in range(n_samples):
            constraints += [loss[:,i] >= demand[:,i] * demand_multiplier - allocation] # original constraint    
        objective = cp.Minimize(cp.sum(loss))
        prob = cp.Problem(objective, constraints)
        prob.solve(cp.GUROBI)
    else:
        print("no trees")
        return None
        
    return allocation.value

# construct demand samples according to LP with tree demand distribution
def optimize_fn_ours(test, train, n_estimators, allocation_max, product):
    demand=[]
    hf_pk=np.array(test['hf_pk'])
    train=train[train['hf_pk'].isin(hf_pk)]
    for i in hf_pk:
        df2=train[(train['hf_pk']==i)]
        data = pd.Series(df2['quantity'])
        #print(hf_pk)
        if len(data)!=0:
            paramSample = st.nakagami.fit(data)
            rng = np.random.default_rng(10)
            dist = st.nakagami
            shapes = paramSample
            data = dist.rvs(*shapes, size=n_estimators, random_state=rng)
            data = np.array(data).T
            demand.append(data)
        else:
            nodata = np.array([0] *n_estimators )
            demand.append(nodata)
    demand=np.array(demand)
    return optimize_lp(demand, allocation_max, product)
    
    
def get_allocation_all(df, n_estimators, allocation_max, optimize_fn,dates):
    
    df['date']=df['date'].map(lambda x:datetime.strptime(x,"%Y-%m-%d"))
    df = df[df['date'] > '2019-02-01']
    df = df.dropna()
    df_allocation = []
    df=df[pd.notnull(df['date'])]
    for product in sorted(df['product'].unique()):
        df_cur = get_allocation(df, n_estimators, allocation_max, product, optimize_fn,dates)
        print(product, evaluate(df_cur))
        df_allocation.append(df_cur)
    return pd.concat(df_allocation, ignore_index = True)

def evaluate(df):
    return np.sum(np.maximum(df['target'] - df['allocation'], 0.0))

def main(args):
    # Step 0: Parameters
    time1=time.time()
    allocation_max = 0 # adjust in get_allocation function
    n_estimators = 500
    dates=list(args.date)
    date_column='date'
    bgt=['AvgHist3']
    target_col='target'
    lead_time=1
    approaches = [('ours', optimize_fn_ours)]
    df = pd.read_csv("/Experiment/df4ML.csv")

    for dateRun in dates: 
        for budgt in bgt: 
            if budgt=="AvgHist3":
                #budget=pd.read_csv(f'tmp/budgetConsump3_2022-12-01.csv')
            elif budgt=="AvgHist1":
                #budget=pd.read_csv(f'tmp/budgetConsump3_2022-12-01.csv')
                budget['stock']=budget['stock']/3
            else:
                budget=pd.read_csv(f'/Experiment/budget2023Q2Script_processed.csv')
                budget=budget[budget['budgetType']==budgt].copy()

            df2=df.merge(budget, on=['product'], how='left')

            for name, optimize_fn in approaches:
                df_allocation = get_allocation_all(df2, n_estimators, allocation_max, optimize_fn,dateRun)
                df_allocation.to_csv(f'/Experiment/Distribution/results/allocationQ2_ModDistribution_{str(dateRun)}_{str(budgt)}.csv')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', nargs='+', type=str, choices=['2023-01-01','2023-02-01','2022-12-01'],default='2023-01-01')
    args = parser.parse_args()
    main(args)

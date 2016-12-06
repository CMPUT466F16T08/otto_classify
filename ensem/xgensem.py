import pandas as pd
import numpy as np
import xgboost as xgb

base=pd.read_csv('../chart/chart.csv')
base=base.values
rid=base[:,0]
label=base[:,-1]
data=base[:,1:-1]

dtrain=xgb.DMatrix( data, label=label)
param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'multi:softmax', 'num_class':9 }

bst = xgb.train(param,dtrain)

dtest=pd.read_csv('../test.csv')
dtest=dtest.values
dtest=dtest[:,1:]

dtest = xgb.DMatrix(dtest)
ypred = bst.predict(dtest)
print ypred

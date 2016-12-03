#format your output according to the sample_output.csv then you can use this script to evaluate it

#don't use test_set to set parameters, split own test_set from train_set to find parameters

import pandas as pd
import numpy as np

X = pd.read_csv('prediction_prob_xgboost.csv')	#file name
y = pd.read_csv('test_set.csv')

X=X.values
X=X[:,1:]

y=y.target.values
onehot=np.zeros(X.shape)
onehot[np.arange(X.shape[0]),y]=1

X=np.maximum(np.minimum(X,1-10**(-15)),10**-15)
result=-1.0/X.shape[0]*np.sum(onehot*np.log(X))

print result

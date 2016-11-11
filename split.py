import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
import numpy as np


X = pd.read_csv('train.csv')
#X = X.drop('id', axis=1)

y = X.target.values
y = LabelEncoder().fit_transform(y)

#X = X.drop('target', axis=1)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=36)

X

Xtrain['target']=ytrain
Xtest['target']=ytest

Xtrain.to_csv('train_set.csv',index=False)
Xtest.to_csv('test_set.csv',index=False)

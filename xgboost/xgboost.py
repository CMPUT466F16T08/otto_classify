
#import xgboost
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss

# read file, get training data
X = pd.read_csv('train_set.csv')

# remove id
X = X.drop('id', axis=1)

# get target and encode
y = X.target.values
y = LabelEncoder().fit_transform(y)

# remove target
X = X.drop('target', axis=1)

# split data
# Reference: http://stackoverflow.com/questions/29438265/stratified-train-test-split-in-scikit-learn
#Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=36)
Xtrain = X
ytrain = y

Xt = pd.read_csv('test_set.csv')

# remove id
Xt = Xt.drop('id', axis=1)

# get target and encode
yt = Xt.target.values
yt = LabelEncoder().fit_transform(yt)

# remove target
Xt = Xt.drop('target', axis=1)

Xtest = Xt
ytest = yt

lr = 0.7#learning_rate_list[0]
md = 15#max_depth_list[0]
ne = 500#n_estimators_list[i]
gbm = GradientBoostingClassifier(learning_rate=lr, max_depth=md, n_estimators=ne)
model = gbm.fit(Xtrain, ytrain)
pred_prob = model.predict_proba(Xtest)

pred_clas = model.predict(Xtest)
logloss = log_loss(ytest, pred_prob)
print 'n_estimators =', ne, 'learning_rate =', lr, 'max_depth =', md, 'log loss =', logloss

'''
learning_rate_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
max_depth_list = [3,5,10]
n_estimators_list = [10, 20, 50, 100, 200]

for i in range(len(n_estimators_list)):
    lr = learning_rate_list[0]
    md = max_depth_list[0]
    ne = n_estimators_list[i]
    gbm = GradientBoostingClassifier(learning_rate=lr, max_depth=md, n_estimators=ne)
    model = gbm.fit(Xtrain, ytrain)
    pred_prob = model.predict_proba(Xtest)
    
    pred_clas = model.predict(Xtest)
    logloss = log_loss(ytest, pred_prob)
    print 'n_estimators =', ne, 'learning_rate =', lr, 'max_depth =', md, 'log loss =', logloss
    
for i in range(len(learning_rate_list)):
    lr = learning_rate_list[i]
    md = max_depth_list[0]
    ne = n_estimators_list[0]
    gbm = GradientBoostingClassifier(learning_rate=lr, max_depth=md, n_estimators=ne)
    model = gbm.fit(Xtrain, ytrain)
    pred_prob = model.predict_proba(Xtest)
    
    pred_clas = model.predict(Xtest)
    logloss = log_loss(ytest, pred_prob)
    print 'n_estimators =', ne, 'learning_rate =', lr, 'max_depth =', md, 'log loss =', logloss
    
for i in range(len(max_depth_list)):
    lr = learning_rate_list[0]
    md = max_depth_list[i]
    ne = n_estimators_list[0]
    gbm = GradientBoostingClassifier(learning_rate=lr, max_depth=md, n_estimators=ne)
    model = gbm.fit(Xtrain, ytrain)
    pred_prob = model.predict_proba(Xtest)
    
    pred_clas = model.predict(Xtest)
    logloss = log_loss(ytest, pred_prob)
    print 'n_estimators =', ne, 'learning_rate =', lr, 'max_depth =', md, 'log loss =', logloss
'''       


'''
n_estimators = 100 learning_rate = 0.5 max_depth = 5 : log loss = 0.687561242769
n_estimators = 100 learning_rate = 0.48 max_depth = 1 log loss = 0.651815005376
n_estimators = 100 learning_rate = 0.45 max_depth = 3: log loss = 0.623512266363
n_estimators = 100 learning_rate = 0.5 max_depth = 2 : log loss = 0.614801832018
n_estimators = 100 learning_rate = 0.48 max_depth = 3 log loss = 0.600838380242
n_estimators = 100 learning_rate = 0.49 max_depth = 2 log loss = 0.600396528478
n_estimators = 100 learning_rate = 0.495 max_depth = 3 log loss = 0.596102075079
n_estimators = 100 learning_rate = 0.475 max_depth = 3 log loss = 0.591128964187
n_estimators = 100 learning_rate = 0.475 max_depth = 2 log loss = 0.583600418683
n_estimators = 100 learning_rate = 0.48 max_depth = 2 log loss = 0.581117435124
n_estimators = 100 learning_rate = 0.5 max_depth = 3 : log loss = 0.574187134015

'''
#import xgboost
import pandas as pd
import numpy as np
import sklearn
import time
import csv
from math import log
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.metrics import log_loss

def write_pred_prob(probs):
    
    id_f = open('../test_set.csv', 'rb')
    id_r = csv.reader(id_f)
    ids = [row[0] for row in id_r]
    ids = ids[1:]
    id_f.close()
    
    f = open('prob.csv', 'wb')
    writer = csv.writer(f)
    labels = ['id']
    for i in range(9):
        labels.append('Class_'+str(i+1))
    writer.writerow(labels)
    data = []
    for l in range(len(probs)):
        new = [ids[l]]
	
        new += probs[l]
        data.append(new)
    writer.writerows(data)
    f.close()
    print 'finish writting <prob.csv>'

def log_loss_implement(actual, predicted, eps = 1e-15):
    predicted = np.minimum(np.maximum(predicted,eps),1-eps)  
    sum1 = 0
    N = len(actual)
    M = num_labels(actual)
    
    result_list = []
    
    for j in range(M):
        sum2 = 0
        count = 0
        for i in range(N):
            y = 1 if j==actual[i] else 0
            if j==actual[i]:
                y = 1
                count += 1
            else:
                y = 0
            p = predicted[i][j]
            temp = y*log(p)
            sum2 += temp
        cla_logloss = (-1)*sum2/float(count)
        print 'Class', j, 'log loss =', cla_logloss
        result_list.append([j, cla_logloss])
        
        sum1 += sum2
    logloss = (-1)*sum1/float(N)
    return logloss, result_list

def num_labels(actual):
    labels = {}
    size = 0
    for l in actual:
        if l not in labels:
            size += 1
            labels[l] = 0
    return size

def write_pred_logloss(logloss_list):
    f = open('logloss.csv', 'wb')
    writer = csv.writer(f)
    labels = ['class', 'log loss']
    writer.writerow(labels)
    data = []
    for l in logloss_list:
        data.append(l)
    writer.writerows(data)
    f.close()
    print 'finish writting <logloss.csv>'    

# read file, get training data
X = pd.read_csv('../train_set.csv')

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

Xt = pd.read_csv('../test_set.csv')

# remove id
Xt = Xt.drop('id', axis=1)

# get target and encode
yt = Xt.target.values
yt = LabelEncoder().fit_transform(yt)

# remove target
Xt = Xt.drop('target', axis=1)

Xtest = Xt
ytest = yt

lr = 0.5#learning_rate_list[0]
md = 3  #max_depth_list[0]
ne = 100#n_estimators_list[i]
gbm = GradientBoostingClassifier(learning_rate=lr, max_depth=md, n_estimators=ne)
gbm = BaggingClassifier(gbm, n_estimators=5)
model = gbm.fit(Xtrain, ytrain)
pred_prob = model.predict_proba(Xtest)

pred_clas = model.predict(Xtest)
logloss = log_loss(ytest, pred_prob)
logloss2, class_logloss = log_loss_implement(ytest, pred_prob)
write_pred_prob(pred_prob)
write_pred_logloss(class_logloss)

print 'n_estimators =', ne, 'learning_rate =', lr, 'max_depth =', md, 'log loss =', logloss
print 'logloss implement =', logloss2

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


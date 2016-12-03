from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import time
import csv
from math import log
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.grid_search import GridSearchCV

'''
k = 100, logloss = 0.736984492464
k = 200, logloss = 0.718084608314
k = 180, logloss = 0.712954462107
k = 150, logloss = 0.712551759434
k = 145, logloss = 0.710122820778
k = 140, logloss = 0.708016821843
'''

def write_pred_prob(probs, C_, g):
    
    id_f = open('../test_set.csv', 'rb')
    id_r = csv.reader(id_f)
    ids = [row[0] for row in id_r]
    ids = ids[1:]
    id_f.close()
    
    f = open(str(C_)+'_'+str(g)+'prob.csv', 'wb')
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

def write_pred_logloss(logloss_list, C_, g):
    f = open(str(C_)+'_'+str(g)+'logloss.csv', 'wb')
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

n_neigh_list = [140, 142, 147]
for n in n_neigh_list:
    neigh = KNeighborsClassifier(n_neighbors=n, n_jobs = -1)
    neigh = neigh.fit(Xtrain, ytrain)
    
    predict_prob = neigh.predict_proba(Xtest)
    logloss3 = log_loss(ytest, predict_prob)
    logloss2, logloss_list = log_loss_implement(ytest, predict_prob)
    
    #write_pred_prob(predict_prob, 0, 0)
    #write_pred_logloss(logloss_list, 0, 0) 
    print '=======number of neighbor =', n,'============'
    print 'logloss implement =', logloss2
    print 'logloss =', logloss3
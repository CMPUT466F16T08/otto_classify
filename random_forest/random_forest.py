import pandas as pd
import numpy as np
import sklearn
import time
import csv
from math import log
#from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

'''
n_estimators=50:

Logloss (with calibration using sigmoid) = 0.535278464876
Accuracy (with calibration using sigmoid) = 0.817307692308

Logloss (with calibration using isotonic) = 0.498130403807
Accuracy (with calibration using isotonic) = 0.816984486102
'''

def predit_result(array):
    choice = []
    for i in range(len(array)):
        line = array[i]
        max_ind = 0
        for j in range(len(line)):
            if float(line[j]) > float(line[max_ind]):
                max_ind = j
        choice.append(max_ind)
    return choice

def cal_accuracy(predict, ytest):
    correct = 0
    for i in range(len(predict)):
        if predict[i]==ytest[i]:
            correct += 1
    return float(correct)/len(predict)  

def class_accuracy(predict, ytest):
    classes = {}
    for i in range(len(predict)):
        c = ytest[i]
        if c not in classes:
            classes[c] = {'correct':0,
                          'false':0}
        if c == predict[i]:
            classes[c]['correct'] += 1
        else:
            classes[c]['false'] += 1
        
    for cla in classes:
        acc = float(classes[cla]['correct'])/(classes[cla]['false'] + classes[cla]['correct'])
        print 'Class', str(cla), 'accuracy =', str(acc)
  
def num_labels(actual):
    labels = {}
    size = 0
    for l in actual:
        if l not in labels:
            size += 1
            labels[l] = 0
    return size

# formula: http://www.exegetic.biz/blog/2015/12/making-sense-logarithmic-loss/
def log_loss_implement(actual, predicted, eps = 1e-15):
    '''
    predicted = np.minimum(np.maximum(predicted,eps),1-eps)  
    sum1 = 0
    N = len(actual)
    M = num_labels(actual)
    for i in range(len(actual)):
        sum2 = 0
        for j in range(M):
            y = 1 if j==actual[i] else 0
            p = predicted[i][j]
            temp = y*log(p)
            sum2 += temp
        sum1 += sum2

    return (-1)*sum1/float(N)
    '''
    
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


def write_pred_prob(probs):
    id_f = open('test_set.csv', 'rb')
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
X = pd.read_csv('train_set.csv')

# remove id
X = X.drop('id', axis=1)

# get target and encode
y = X.target.values
y = LabelEncoder().fit_transform(y)
# remove target
X = X.drop('target', axis=1)


Xt = pd.read_csv('test_set.csv')

# remove id
Xt = Xt.drop('id', axis=1)

# get target and encode
yt = Xt.target.values
yt = LabelEncoder().fit_transform(yt)

# remove target
Xt = Xt.drop('target', axis=1)

Xtrain = X
ytrain = y

Xtest = Xt
ytest = yt



'''
Feature importance
http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
'''
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(Xtrain, ytrain)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
print importances
print '\n'
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

'''
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
'''
'''
Calibration
'''
cla_imp = {}
for i in range(len(importances)):
    cla_imp[i+1] = importances[i]
    
cv_list = [2, 3, 4, 5, 6, 7, 8]
for c in cv_list:
    start = time.time()
    forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_features='sqrt', min_samples_leaf=1, class_weight='balanced')
    forestCal2 = CalibratedClassifierCV(forest, method='isotonic', cv=c)
    forestCal2 = forestCal2.fit(Xtrain, ytrain)
    predict_prob2 = forestCal2.predict_proba(Xtest) 
    end = time.time()
    run_time = end - start
    #prediction = predit_result(predict_prob2)
    #acc_class = class_accuracy(prediction, ytest)
    logloss = log_loss(ytest, predict_prob2)
    logloss2, logloss_list = log_loss_implement(ytest, predict_prob2)
    #accuracy = cal_accuracy(prediction, ytest)
    print 'cv =', c
    print '\nLogloss (with calibration using isotonic) = ' + str(logloss2) + ' compare:', str(logloss)
    #print 'Accuracy (with calibration using isotonic) = ' + str(accuracy)
    print 'time =', run_time

#write_pred_prob(predict_prob2)
#write_pred_logloss(logloss_list)

'''
forestCal = CalibratedClassifierCV(forest, method='sigmoid', cv=5)
forestCal = forestCal.fit(Xtrain, ytrain)
predict_prob = forestCal.predict_proba(Xtest)
logloss = log_loss(ytest, predict_prob)
prediction = predit_result(predict_prob)
accuracy = cal_accuracy(prediction, ytest)
#scores = metrics.accuracy_score(prediction, ytest)
print '\nLogloss (with calibration using sigmoid) = ' + str(logloss)
print 'Accuracy (with calibration using sigmoid) = ' + str(accuracy)
#print 'score =', str(scores)
'''

'''
Choosing parameter
'''
'''
# choosing n_estimators here:
num_est_list = [10,20,30,40,50,100,250,500]
for ne in num_est_list:
    start = time.clock()
    forest = RandomForestClassifier(n_estimators=ne, n_jobs=-1, max_features='sqrt')
    forest = forest.fit(Xtrain, ytrain)
    predict = forest.predict(Xtest)
    end = time.clock()
    run_time = end - start
    correct = 0
    for i in range(len(predict)):
        if predict[i]==ytest[i]:
            correct += 1
    print 'n_estimators = '+str(ne)
    print '   |-> accuracy = ' + str(float(correct)/len(predict))
    print '   |-> running time = ' + str(run_time)
    
# use n_estimators = 100
'''
'''
# choosing min_samples_leaf here:
num_leaf_list = [1,2,3,5,7,10,20,30,40,50,100]
for nl in num_leaf_list:
    start = time.clock()
    forest = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_features='sqrt', min_samples_leaf=nl)
    forest = forest.fit(Xtrain, ytrain)
    predict = forest.predict(Xtest)
    end = time.clock()
    run_time = end - start
    correct = 0
    for i in range(len(predict)):
        if predict[i]==ytest[i]:
            correct += 1
    print 'min_sample_leaf = '+str(nl)
    print '   |-> accuracy = ' + str(float(correct)/len(predict))
    print '   |-> running time = ' + str(run_time)
# use min_samples_leaf = 1
'''

'''
without calibration:
'''
'''
forestBag = BaggingClassifier(forest, n_estimators=5)
forestBag = forestBag.fit(Xtrain, ytrain)
predict_prob0 = forestBag.predict_proba(Xtest)
logloss0 = log_loss(ytest, predict_prob0)
prediction = predit_result(predict_prob0)
accuracy = cal_accuracy(prediction, ytest)
print '\nLogloss (without calibration) = ' + str(logloss0)
print 'Accuracy (without calibration) = ' + str(accuracy)
'''

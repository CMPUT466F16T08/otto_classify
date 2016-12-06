import pandas as pd
import numpy as np
import sklearn
import time
import csv
import cPickle as pickle
from math import log
#from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

'''
Reference: http://mlwave.com/kaggle-ensembling-guide/
    1.Split the training set into two disjoint sets.
    2.Train several base learners on the first part.
    3.Test the base learners on the second part.
    4.Using the predictions from 3) as the inputs, and the correct responses as the outputs, train a higher level learner.
'''
'''
    Class 0 log loss = 1.81006222877
    Class 1 log loss = 0.500747771575
    Class 2 log loss = 1.18498811702
    Class 3 log loss = 1.94049107604
    Class 4 log loss = 0.329764348788
    Class 5 log loss = 0.285778343501
    Class 6 log loss = 1.44070043069
    Class 7 log loss = 0.404446699186
    Class 8 log loss = 0.560933894566
    
    RF: Logloss (with calibration using isotonic) = 0.670662403093
'''

#from useful_functions import predit_result, cal_accuracy, class_accuracy, num_labels, log_loss_implement, write_pred_prob, write_pred_logloss, init_set, update_set
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


def write_pred_prob(probs,filename):
    ids=pd.read_csv('../test.csv')
    ids=ids.values
    ids=ids[:,0]
    
    probs=np.insert(probs,0,ids,axis=1)
    rpy=pd.DataFrame(probs,columns=['id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'])
    rpy.to_csv(filename,index=False)
    
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
    
def init_set(prob):
    new_set = []
    for i in range(len(prob)):
        new_set.append([])
        for j in range(len(prob[i])):
                new_set[i].append(1e-15)
            
    f = open('new_set.csv', 'wb')
    writer = csv.writer(f)  
    for l in new_set:
        writer.writerow(l)
    f.close()
    
    return new_set

def update_set(new_set, prob):
    for i in range(len(prob)):
        largest = max(prob[i])
        for j in range(len(prob[i])):
            if prob[i][j] == largest:
                new_set[i][j] += prob[i][j]
            
    f = open('new_set.csv', 'wb')
    writer = csv.writer(f)  
    for l in new_set:
        writer.writerow(l)
    f.close()   
    
    return new_set

#====================knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV

def knn(Xtrain, ytrain):
    neigh = KNeighborsClassifier(n_neighbors=142, n_jobs = -1)
    neigh = neigh.fit(Xtrain, ytrain)
    
    predict_prob = neigh.predict_proba(Xtrain)
    return predict_prob, neigh

#====================LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV

def lda(Xtrain, ytrain):
    clf=LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                  solver='svd', store_covariance=False, tol=0.0001)
    clf_CV= CalibratedClassifierCV(clf, method='isotonic', cv=5)
    clf_fit_CV = clf_CV.fit(Xtrain, ytrain)
    predict_prob= clf_fit_CV.predict_proba(Xtrain)
    return predict_prob, clf_fit_CV

#===================NN
import os.path
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def nn(Xtrain, ytrain):
    mlp = MLPClassifier(solver='sgd', learning_rate='constant',momentum=0,alpha=1e-5,
               learning_rate_init=0.2,max_iter=15,verbose=False,random_state=0)
    mlp = mlp.fit(Xtrain, ytrain)
    predict_prob = mlp.predict_proba(Xtrain)    
    return predict_prob, mlp

#===================RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

def rf(Xtrain, ytrain):
    forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_features='sqrt', min_samples_leaf=1, class_weight='balanced')
    forestCal = CalibratedClassifierCV(forest, method='isotonic', cv=6)
    forestCal = forestCal.fit(Xtrain, ytrain)
    predict_prob = forestCal.predict_proba(Xtrain)
    return predict_prob, forestCal

#==================SVM
from sklearn import svm

def svm_(Xtrain, ytrain):
    model = svm.SVC(kernel='rbf', class_weight='balanced', C=2, gamma=0.003, probability=True)
    model = model.fit(Xtrain, ytrain)
    predict_prob = model.predict_proba(Xtrain)
    return predict_prob, model
    
#==================XGB
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier

def xgboost(Xtrain, ytrain):
    gbm = GradientBoostingClassifier(learning_rate=0.4, max_depth=3, n_estimators=100)
    gbm = BaggingClassifier(gbm, n_estimators=5)
    model = gbm.fit(Xtrain, ytrain)
    predict_prob = model.predict_proba(Xtrain)    
    return predict_prob, model
    


'''
read and test
'''

# read file, get training data and testing data
X = pd.read_csv('../train.csv')
X = X.drop('id', axis=1)
y = X.target.values
y = LabelEncoder().fit_transform(y)
X = X.drop('target', axis=1)

Xt = pd.read_csv('../test.csv')
Xt = Xt.drop('id', axis=1)
#yt = Xt.target.values
#yt = LabelEncoder().fit_transform(yt)
#Xt = Xt.drop('target', axis=1)

Xtrain1 = X
ytrain1 = y


Xtest = Xt
#ytest = yt


#first step

print 'First step'

prob, knn_model = knn(Xtrain1, ytrain1)
#new_set = init_set(prob)
#new_set = update_set(new_set, prob)
print 'knn'

prob, lda_model = lda(Xtrain1, ytrain1)
#new_set = update_set(new_set, prob)
print 'lda'

prob, nn_model = nn(Xtrain1, ytrain1)
#new_set = update_set(new_set, prob)
print 'nn'

prob, rf_model = rf(Xtrain1, ytrain1)
#new_set = update_set(new_set, prob)
print 'rf'

prob, svm_model = svm_(Xtrain1, ytrain1)
#new_set = update_set(new_set, prob)
print 'svm'

prob, xgb_model = xgboost(Xtrain1, ytrain1)
#new_set = update_set(new_set, prob)
print 'xgboost'

'''
second step
'''
print 'Second step'
'''
id_f = open('new_set.csv', 'rb')
id_r = csv.reader(id_f)
Xtrain = [row for row in id_r]
id_f.close()
'''
#======================================
prob = knn_model.predict_proba(Xtest)
write_pred_prob(prob, 'knn_prob.csv')
#new_set = init_set(prob)
#new_set = update_set(new_set, prob)

prob = lda_model.predict_proba(Xtest)
write_pred_prob(prob, 'lda_prob.csv')
#new_set = update_set(new_set, prob)

prob = nn_model.predict_proba(Xtest)
write_pred_prob(prob, 'nn_prob.csv')
#new_set = update_set(new_set, prob)

prob = rf_model.predict_proba(Xtest)
write_pred_prob(prob, 'rf_prob.csv')
#new_set = update_set(new_set, prob)

prob = svm_model.predict_proba(Xtest)
write_pred_prob(prob, 'svm_prob.csv')
#new_set = update_set(new_set, prob)

prob = xgb_model.predict_proba(Xtest)
write_pred_prob(prob, 'xgb_prob.csv')
#new_set = update_set(new_set, prob)


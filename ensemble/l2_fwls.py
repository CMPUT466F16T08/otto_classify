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
RF: 
    only deal with the highest probability, not use svm in test set
    Class 0 log loss = 1.99367278161
    Class 1 log loss = 1.21382153481
    Class 2 log loss = 0.555585076175 ***
    Class 3 log loss = 8.12407268906
    Class 4 log loss = 0.339428380242
    Class 5 log loss = 0.241941270692
    Class 6 log loss = 1.70264027549
    Class 7 log loss = 0.395128899931
    Class 8 log loss = 0.664684844387
    Logloss (with calibration using isotonic) = 1.06729834639
'''
'''

Class 0 log loss = 4.63578176929
Class 1 log loss = 0.544965377405
Class 2 log loss = 3.03881088369
Class 3 log loss = 15.7513858174
Class 4 log loss = 0.340182931615
Class 5 log loss = 0.393712675862
Class 6 log loss = 2.68731498249
Class 7 log loss = 0.389444521871
Class 8 log loss = 0.659284676578

RF: Logloss (with calibration using isotonic) = 1.70978982541
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
        #print 'Class', j, 'log loss =', cla_logloss
        result_list.append([j, cla_logloss])
        
        sum1 += sum2
    logloss = (-1)*sum1/float(N)
    return logloss, result_list


def write_pred_prob(probs, filename):
    id_f = open(filename, 'rb')
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


def cal_feature_weight(V, alg, cla_logloss):
    V[alg] = []
    for ll in cla_logloss:
        V[alg].append(logloss/float(ll[1]))
    return V
    

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
X = pd.read_csv('../train_set.csv')
X = X.drop('id', axis=1)
y = X.target.values
y = LabelEncoder().fit_transform(y)
X = X.drop('target', axis=1)

Xt = pd.read_csv('../test_set.csv')
Xt = Xt.drop('id', axis=1)
yt = Xt.target.values
yt = LabelEncoder().fit_transform(yt)
Xt = Xt.drop('target', axis=1)

Xtrain = X
ytrain = y

Xtest = Xt
ytest = yt


#first step

print 'First step'

prob, knn_model = knn(Xtrain, ytrain)
new_set = init_set(prob)
new_set = update_set(new_set, prob)
print 'knn'
print 'saving models ...'
f = open("knn_model.pkl", "wb")
pickle.dump(knn_model, f, protocol=2)
f.close()

prob, lda_model = lda(Xtrain, ytrain)
new_set = update_set(new_set, prob)
print 'lda'
print 'saving models ...'
f = open("lda_model.pkl", "wb")
pickle.dump(lda_model, f, protocol=2)
f.close()

prob, nn_model = nn(Xtrain, ytrain)
new_set = update_set(new_set, prob)
print 'nn'
print 'saving models ...'
f = open("nn_model.pkl", "wb")
pickle.dump(nn_model, f, protocol=2)
f.close()

prob, rf_model = rf(Xtrain, ytrain)
new_set = update_set(new_set, prob)
print 'rf'
f = open("rf_model.pkl", "wb")
pickle.dump(rf_model, f, protocol=2)
f.close()

prob, svm_model = svm_(Xtrain, ytrain)
new_set = update_set(new_set, prob)
print 'svm'
f = open("svm_model.pkl", "wb")
pickle.dump(svm_model, f, protocol=2)
f.close()

prob, xgb_model = xgboost(Xtrain, ytrain)
#new_set = update_set(new_set, prob)
print 'xgboost'
f = open("xgb_model.pkl", "wb")
pickle.dump(xgb_model, f, protocol=2)
f.close()


'''
second step
'''
'''
print 'Second step'

print 'loading models ...'

f = open("knn_model.pkl", "rb")
knn_model = pickle.load(f)
f.close()
print 1
f = open("lda_model.pkl", "rb")
lda_model = pickle.load(f)
f.close()
print 2
f = open("nn_model.pkl", "rb")
nn_model = pickle.load(f)
f.close()
print 3
f = open("rf_model.pkl", "rb")
rf_model = pickle.load(f)
f.close()
print 4
f = open("svm_model.pkl", "rb")
svm_model = pickle.load(f)
f.close()
print 5
f = open("xgb_model.pkl", "rb")
xgb_model = pickle.load(f)
f.close()
print 6

# weight for each method
V = {}
# probs for each method

prob = knn_model.predict_proba(Xtest)
logloss, cla_logloss = log_loss_implement(ytest, prob)
V = cal_feature_weight(V, 'knn', cla_logloss)

prob = lda_model.predict_proba(Xtest)
logloss, cla_logloss = log_loss_implement(ytest, prob)
V = cal_feature_weight(V, 'lda', cla_logloss)

prob = nn_model.predict_proba(Xtest)
logloss, cla_logloss = log_loss_implement(ytest, prob)
V = cal_feature_weight(V, 'nn', cla_logloss)

prob = rf_model.predict_proba(Xtest)
logloss, cla_logloss = log_loss_implement(ytest, prob)
V = cal_feature_weight(V, 'rf', cla_logloss)

prob = svm_model.predict_proba(Xtest)
logloss, cla_logloss = log_loss_implement(ytest, prob)
V = cal_feature_weight(V, 'svm', cla_logloss)

prob = xgb_model.predict_proba(Xtest)
logloss, cla_logloss = log_loss_implement(ytest, prob)
V = cal_feature_weight(V, 'xgb', cla_logloss)

X = pd.read_csv('test.csv')
X = X.drop('id', axis=1)
#y = X.target.values
#y = LabelEncoder().fit_transform(y)
#X = X.drop('target', axis=1)

Xtest = X
ytest = y

P = {}
prob = knn_model.predict_proba(Xtest)
P['knn'] = prob
prob = lda_model.predict_proba(Xtest)
P['lda'] = prob
prob = nn_model.predict_proba(Xtest)
P['nn'] = prob
prob = rf_model.predict_proba(Xtest)
P['rf'] = prob
prob = svm_model.predict_proba(Xtest)
P['svm'] = prob
prob = xgb_model.predict_proba(Xtest)
P['xgb'] = prob

new_set = []
for i in range(len(prob)):
    result = [1e-15]*len(prob[0])
    for alg,weight in V:
        for j in range(len(weight)):
            result[j] = weight[j]*prob[i][j]
    new_set.append(result)

write_pred_prob(new_set, 'test.csv')
'''
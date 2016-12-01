import pandas as pd
import numpy as np
import sklearn
import time
from math import log
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
# Reference: http://blog.csdn.net/zouxy09/article/details/48903179
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn import metrics

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
        
def class_logloss(prediction, ytest, predict_prob):
    classes = {}
    for i in range(len(ytest)):
        c = ytest[i]
        if c not in classes:
            classes[c] = [predict_prob[i]]
        else:
            classes[c].append(predict_prob[i])

    for cla in classes: 
        #print 'Class', cla, 'size =', len(classes[cla])
        correct = [cla]*len(classes[cla])
        cla_logloss = log_loss_implement(correct, classes[cla])
        print 'Class', str(cla), 'log loss =', str(cla_logloss)
        
    '''
    cla = 1
    print 'Class', cla, 'size =', len(classes[cla])
    correct = [cla]*len(classes[cla])
    cla_logloss = log_loss_implement(correct, classes[cla])
    print 'Class', str(cla), 'log loss =', str(cla_logloss)    
    '''
def num_labels(actual):
    labels = {}
    size = 0
    for l in actual:
        if l not in labels:
            size += 1
            labels[l] = 0
    return size

# http://www.exegetic.biz/blog/2015/12/making-sense-logarithmic-loss/
def log_loss_implement(actual, predicted, eps = 1e-15):
    predicted = np.minimum(np.maximum(predicted,eps),1-eps)  
    print len(actual)
    sum1 = 0
    N = len(actual)
    for i in range(len(actual)):
        sum2 = 0
        M = num_labels(actual)
        for j in range(M):
            y = 1 if j==actual[i] else 0
            p = predicted[i][j]
            temp = y*log(p)
            sum2 += temp
        sum1 += sum2

    return (-1)*sum1/float(N)
    
# read file, get training data
X = pd.read_csv('train.csv')

# remove id
X = X.drop('id', axis=1)

# get target and encode
y = X.target.values
y = LabelEncoder().fit_transform(y)

# remove target
X = X.drop('target', axis=1)

# split data
# Reference: http://stackoverflow.com/questions/29438265/stratified-train-test-split-in-scikit-learn
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=36)

forest = RandomForestClassifier(n_estimators=10, n_jobs=-1, max_features='sqrt', min_samples_leaf=1)

forestCal2 = CalibratedClassifierCV(forest, method='isotonic', cv=5)
forestCal2 = forestCal2.fit(Xtrain, ytrain)
predict_prob2 = forestCal2.predict_proba(Xtest)
logloss3 = log_loss(ytest, predict_prob2)
logloss2 = log_loss_implement(ytest, predict_prob2)

prediction = predit_result(predict_prob2)
acc_class = class_accuracy(prediction, ytest)
logl_class = class_logloss(prediction, ytest, predict_prob2)
accuracy = cal_accuracy(prediction, ytest)
print '\nLogloss (with calibration using isotonic) = ' + str(logloss2) + ' compare:', str(logloss3)
print 'Accuracy (with calibration using isotonic) = ' + str(accuracy)

'''
forestCal = CalibratedClassifierCV(forest, method='sigmoid', cv=5)
forestCal = forestCal.fit(Xtrain, ytrain)
predict_prob = forestCal.predict_proba(Xtest)
logloss = log_loss(ytest, predict_prob)
prediction = predit_result(predict_prob)
accuracy = cal_accuracy(prediction, ytest)
scores = metrics.accuracy_score(prediction, ytest)
print '\nLogloss (with calibration using sigmoid) = ' + str(logloss)
print 'Accuracy (with calibration using sigmoid) = ' + str(accuracy)
print 'score =', str(scores)
'''

'''
forest2 = GradientBoostingClassifier(n_estimators=50)

forestCal = forest2.fit(Xtrain, ytrain)
predict_prob = forestCal.predict_proba(Xtest)
logloss = log_loss(ytest, predict_prob)
prediction = predit_result(predict_prob)
accuracy = cal_accuracy(prediction, ytest)
print '\nLogloss (gradient boosting) = ' + str(logloss)
print 'Accuracy (gradient boosting) = ' + str(accuracy)


forestCal = CalibratedClassifierCV(forest2, method='sigmoid', cv=5)
forestCal = forestCal.fit(Xtrain, ytrain)
predict_prob = forestCal.predict_proba(Xtest)
logloss = log_loss(ytest, predict_prob)
prediction = predit_result(predict_prob)
accuracy = cal_accuracy(prediction, ytest)
print '\nLogloss (gradient boosting) = ' + str(logloss)
print 'Accuracy (gradient boosting) = ' + str(accuracy)
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
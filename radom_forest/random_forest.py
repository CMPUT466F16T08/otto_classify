import pandas as pd
import sklearn
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

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

forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_features='sqrt', min_samples_leaf=1)

forestCal = CalibratedClassifierCV(forest, method='sigmoid', cv=5)
forestCal = forestCal.fit(Xtrain, ytrain)
predict_prob = forestCal.predict_proba(Xtest)
logloss = log_loss(ytest, predict_prob)
prediction = predit_result(predict_prob)
accuracy = cal_accuracy(prediction, ytest)
print '\nLogloss (with calibration using sigmoid) = ' + str(logloss)
print 'Accuracy (with calibration using sigmoid) = ' + str(accuracy)

forestCal2 = CalibratedClassifierCV(forest, method='isotonic', cv=5)
forestCal2 = forestCal2.fit(Xtrain, ytrain)
predict_prob2 = forestCal2.predict_proba(Xtest)
logloss2 = log_loss(ytest, predict_prob2)
prediction = predit_result(predict_prob2)
accuracy = cal_accuracy(prediction, ytest)
print '\nLogloss (with calibration using isotonic) = ' + str(logloss2)
print 'Accuracy (with calibration using isotonic) = ' + str(accuracy)

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
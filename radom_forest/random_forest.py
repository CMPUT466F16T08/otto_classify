import pandas as pd
import sklearn
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV

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

# Parameters: 
#     Classification: 
#         n_estimators = 100
#         min_samples_leaf = 1
#
#     Calibration: 
#         method = 'sigmoid'

forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_features='sqrt', min_samples_leaf=1)
forestCal = CalibratedClassifierCV(forest, method='sigmoid', cv=5)
forestCal = forestCal.fit(Xtrain, ytrain)
predict = forestCal.predict(Xtest)
correct = 0
for i in range(len(predict)):
    if predict[i]==ytest[i]:
        correct += 1
print '\nAccuracy (with calibration using sigmoid) = ' + str(float(correct)/len(predict))
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
forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_features='sqrt', min_samples_leaf=1)
forest = forest.fit(Xtrain, ytrain)
predict = forest.predict(Xtest)
correct = 0
for i in range(len(predict)):
    if predict[i]==ytest[i]:
        correct += 1
print 'Accuracy /without bagging /without calibration= ' + str(float(correct)/len(predict))

forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_features='sqrt', min_samples_leaf=1)
forestBag = BaggingClassifier(forest, n_estimators=5)
forestBag = forestBag.fit(Xtrain, ytrain)
predict = forestBag.predict(Xtest)
correct = 0
for i in range(len(predict)):
    if predict[i]==ytest[i]:
        correct += 1
print '\nAccuracy /with bagging /without calibration = ' + str(float(correct)/len(predict))

forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_features='sqrt', min_samples_leaf=1)
forestCal = CalibratedClassifierCV(forest, method='isotonic', cv=5)
forestCal = forestCal.fit(Xtrain, ytrain)
predict = forestCal.predict(Xtest)
correct = 0
for i in range(len(predict)):
    if predict[i]==ytest[i]:
        correct += 1
print '\nAccuracy /with calibration using isotonic = ' + str(float(correct)/len(predict))

forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_features='sqrt', min_samples_leaf=1)
forestCal = CalibratedClassifierCV(forest, method='sigmoid', cv=5)
forestCal = forestCal.fit(Xtrain, ytrain)
predict = forestCal.predict(Xtest)
correct = 0
for i in range(len(predict)):
    if predict[i]==ytest[i]:
        correct += 1
print '\nAccuracy /with calibration using sigmoid = ' + str(float(correct)/len(predict))
'''

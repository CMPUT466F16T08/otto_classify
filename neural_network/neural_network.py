import os.path
import sys
import sklearn
import pandas as pd
import numpy as np
import time
import csv
from math import log
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler


#http://scikit-learn.org/stable/modules/neural_networks_supervised.html
# Train model and make predictions
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
    print 'finish writting <prob_neural_network.csv>'
def num_labels(actual):
    labels = {}
    size = 0
    for l in actual:
        if l not in labels:
            size += 1
            labels[l] = 0
    return size

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
        print 'Class', str(cla+1), 'accuracy =', str(acc)
#Following code from Han Wang
X = pd.read_csv('train_set.csv')
X1=pd.read_csv('test_set.csv')

# remove id
X = X.drop('id', axis=1)
X1 = X1.drop('id', axis=1)

# get target and encode
ytrain= X.target.values
ytrain= LabelEncoder().fit_transform(ytrain)
ytest= X1.target.values
ytest= LabelEncoder().fit_transform(ytest)
# remove target
Xtrain= X.drop('target', axis=1)
Xtest= X1.drop('target', axis=1)

# split data
# Reference: http://stackoverflow.com/questions/29438265/stratified-train-test-split-in-scikit-learn
scaler = StandardScaler()  
# Don't cheat - fit only on training data
scaler.fit(Xtrain)  
Xtrain = scaler.transform(Xtrain)  
# apply same transformation to test data
Xtest = scaler.transform(Xtest)  

#http://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html#sphx-glr-auto-examples-neural-networks-plot-mlp-training-curves-py
'''params = [{'solver': 'sgd', 'learning_rate': 'constant','momentum': 0,'alpha':1e-6,
           'learning_rate_init': 0.2,'max_iter':15},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,'alpha':1e-5,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2,'max_iter':15},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,'alpha':1e-5,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2,'max_iter':15},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
           'learning_rate_init': 0.2,'max_iter':15},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2,'max_iter':15},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2,'max_iter':15},
          {'solver': 'adam', 'learning_rate_init': 0.01,'hidden_layer_sizes':(100,100),'max_iter':15},
          {'solver': 'lbfgs', 'learning_rate': 'constant','hidden_layer_sizes':(100,100),'momentum': 0,
        	'learning_rate_init': 0.2,'max_iter':100}]

labels = ["constant learning-rate", "constant with momentum",
          "constant with Nesterov's momentum",
          "inv-scaling learning-rate", "inv-scaling with momentum",
          "inv-scaling with Nesterov's momentum", "adam"
          ,"constant learning-rate lbfgs"]
mlps = []
for label, param in zip(labels, params):
		print("training: %s" % label)
		mlp = MLPClassifier(verbose=False,random_state=0, **param)
		mlp.fit(Xtrain, ytrain)
		mlps.append(mlp)
		predict=mlp.predict(Xtest)
		correct = 0
		for i in range(len(predict)):
			if predict[i]==ytest[i]:
				correct += 1
		predict_prob = mlp.predict_proba(Xtest)
		logloss = log_loss(ytest, predict_prob)
		accur=float(correct)/len(predict)
		#acc_class = class_accuracy(predict, ytest)  
		logloss2, logloss_list = log_loss_implement(ytest, predict_prob)
		print("Training set logloss: %f" % logloss)
		'''

mlp = MLPClassifier(solver='sgd', learning_rate='constant',momentum=0,alpha=1e-5,
           learning_rate_init=0.2,max_iter=15,verbose=False,random_state=0)
mlp.fit(Xtrain, ytrain)
predict=mlp.predict(Xtest)
predict_prob = mlp.predict_proba(Xtest)
logloss = log_loss(ytest, predict_prob)
#accur=float(correct)/len(predict)
#acc_class = class_accuracy(predict, ytest)  
logloss2, logloss_list = log_loss_implement(ytest, predict_prob)
print("Training set logloss: %f" % logloss)
write_pred_prob(predict_prob)
write_pred_logloss(logloss_list)
        
'''clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
hidden_layer_sizes=(30, 30,30), random_state=1)
clf.fit(Xtrain, ytrain)
predict=clf.predict(Xtest)
predict_prob = clf.predict_proba(Xtest)
logloss = log_loss(ytest, predict_prob)
print '\nLogloss (with calibration using sigmoid) = ' + str(logloss)

correct = 0
for i in range(len(predict)):
    if predict[i]==ytest[i]:
        correct += 1
print '\nAccuracy (with Neural Network(supervised)) = ' + str(float(correct)/len(predict))'''










import sklearn
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

from sklearn.neural_network import MLPClassifier


#http://scikit-learn.org/stable/modules/neural_networks_supervised.html
# Train model and make predictions
import numpy


Xtrain=numpy.load('Xtrain.npy')
Xtest=numpy.load('Xtest.npy')
ytrain=numpy.load('ytrain.npy')
ytest=numpy.load('ytest.npy')

#Following code from Han Wang
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(Xtrain, ytrain)
predict=clf.predict(Xtest)
correct = 0
for i in range(len(predict)):
    if predict[i]==ytest[i]:
        correct += 1
print '\nAccuracy (with Neural Network(supervised)) = ' + str(float(correct)/len(predict))
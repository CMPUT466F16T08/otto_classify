import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

'''x = pd.read_csv('../train_set.csv')
x=x.values
ylabel= x[:,-1]
x=x[:,1:-1]
y=np.zeros((x.shape[0],9))
y[np.arange(x.shape[0]),ylabel]=1'''


# read file, get training data
X = pd.read_csv('../train.csv')

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


LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
              solver='svd', store_covariance=False, tol=0.0001)

clf = LinearDiscriminantAnalysis()
clf_fit = clf.fit(X, y)
clf_predict= clf_fit.predict(Xtest)

correct=0
for i in range(len(clf_predict)):
    if clf_predict[i]==ytest[i]:
        correct += 1

y_pred= clf.predict_proba(Xtest)
loglossResult=log_loss(ytest,y_pred)
        


print '\nAccuracy (with Linear_Discriminant_Analysis) = ' + str(float(correct)/len(clf_predict))

print '\nThe log loss = ' + str(loglossResult)
'''print 'x is : ', X
print 'Xtest is : ', Xtest
print 'y is : ', y
print 'yTest is : ', ytest
print 'y_pred is: ', y_pred'''

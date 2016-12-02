import pandas as pd
import numpy as np
from function_maybe_useful import *
from sklearn.cross_validation import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV

'''x = pd.read_csv('../train_set.csv')
x=x.values
ylabel= x[:,-1]
x=x[:,1:-1]
y=np.zeros((x.shape[0],9))
y[np.arange(x.shape[0]),ylabel]=1'''




#Accuracy (with Singular value decomposition Linear_Discriminant_Analysis) = 0.704831932773

#The Singular value decomposition log loss = 0.910064928191

#Accuracy (with Least squares solution Linear_Discriminant_Analysis) = 0.704831932773

#The Least squares solution log loss = 0.978570720651

#Accuracy (with Eigenvalue decomposition Least squares solution Linear_Discriminant_Analysis) = 0.689964447317

#The Eigenvalue decomposition log loss = 1.63984890391


#====================================Train Data part=======================
# read file, get training data
#X = pd.read_csv('../train.csv')
X=pd.read_csv('../train_set.csv')

# remove id
X = X.drop('id', axis=1)

# get target and encode
y = X.target.values
y = LabelEncoder().fit_transform(y)

# remove target
X = X.drop('target', axis=1)

# split data
# Reference: http://stackoverflow.com/questions/29438265/stratified-train-test-split-in-scikit-learn
######Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=36)

Xtrain=X
ytrain=y

#============================================================================

#=======================================Test Data part====================
Xt=pd.read_csv('../test_set.csv')

#remove id
Xt=Xt.drop('id',axis=1)

#get target and encode
yt = Xt.target.values
yt = LabelEncoder().fit_transform(yt)

#remove target
Xt=Xt.drop('target',axis=1)

Xtest=Xt
ytest=yt

#===================Singular value decomposition part=================

def SingularValue_LDA():
    
    clf=LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                  solver='svd', store_covariance=False, tol=0.0001)

    clf_CV= CalibratedClassifierCV(clf, method='isotonic', cv=5)
    
    
    #clf_fit = clf.fit(Xtrain, ytrain)

    clf_fit_CV = clf_CV.fit(Xtrain, ytrain)
    
    #======================================
    clf_predict= clf_fit_CV.predict_proba(Xtest)
    
    correct=0
    '''for i in range(len(clf_predict)):
        if clf_predict[i]==ytest[i]:
            correct += 1'''
    
    #y_pred= clf_fit.predict_proba(Xtest)
    #loglossResult=log_loss(ytest,y_pred)

    y_pred_CV= clf_fit_CV.predict_proba(Xtest)
    loglossResult = log_loss(ytest, y_pred_CV)

    loglossResult_withIsotonic, logloss_list=log_loss_implement(ytest, y_pred_CV)
            
    
    
    print '\nAccuracy (with Singular value decomposition Linear_Discriminant_Analysis) = ' + str(float(correct)/len(clf_predict))
    
    #print '\nThe Singular value decomposition log loss = ' + str(loglossResult)

    print '\nSingular value decomposition Logloss (with calibration using isotonic) = ' + str(loglossResult_withIsotonic) + ' compare:', str(loglossResult)

    
    
    for i in range(len(y_pred_CV)):
        for j in range(9):
            y_pred_CV[i][j]=str(y_pred_CV[i][j])

    print '\n instance is : ', y_pred_CV
    write_pred_prob(y_pred_CV)
    

#================================================================

#===================Least squares solution part=================
def lsqr_LDA():
    
    clf=LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                  solver='lsqr', store_covariance=False, tol=0.0001)
    
    #clf = LinearDiscriminantAnalysis()
    clf_fit = clf.fit(X, y)
    clf_predict= clf_fit.predict(Xtest)
    
    '''correct=0
    for i in range(len(clf_predict)):
        if clf_predict[i]==ytest[i]:
            correct += 1'''
    
    y_pred= clf_fit.predict_proba(Xtest)
    loglossResult=log_loss(ytest,y_pred)
            
    
    
    '''print '\nAccuracy (with Least squares solution Linear_Discriminant_Analysis) = ' + str(float(correct)/len(clf_predict))'''
    
    print '\nThe Least squares solution log loss = ' + str(loglossResult)
   
   
#=======================Eigenvalue decomposition part================= 
def eigen_LDA():
    
    clf=LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                      solver='eigen', store_covariance=False, tol=0.0001)
        
    #clf = LinearDiscriminantAnalysis()
    clf_fit = clf.fit(X, y)
    clf_predict= clf_fit.predict(Xtest)
    
    correct=0
    for i in range(len(clf_predict)):
        if clf_predict[i]==ytest[i]:
            correct += 1
    
    y_pred= clf_fit.predict_proba(Xtest)
    loglossResult=log_loss(ytest,y_pred)
            
    
    
    print '\nAccuracy (with Eigenvalue decomposition Least squares solution Linear_Discriminant_Analysis) = ' + str(float(correct)/len(clf_predict))
    
    print '\nThe Eigenvalue decomposition log loss = ' + str(loglossResult)
    
    
    
SingularValue_LDA()
#lsqr_LDA()
#eigen_LDA()

    
'''print 'x is : ', X
print 'Xtest is : ', Xtest
print 'y is : ', y
print 'yTest is : ', ytest
print 'y_pred is: ', y_pred
print 'clf is: ', clf
print 'clf_fit is: ', clf_fit '''

import pandas as pd
import numpy as np
from function_maybe_useful import *
from sklearn.cross_validation import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
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
import matplotlib as mpl
from matplotlib import colors
'''
Singular value Decomposition Result

Class 0 log loss = 2.06941240843
Class 1 log loss = 0.591419469211
Class 2 log loss = 1.30616702463
Class 3 log loss = 2.34475175524
Class 4 log loss = 0.282765340446
Class 5 log loss = 0.240875351478
Class 6 log loss = 1.53522373041
Class 7 log loss = 0.513180995727
Class 8 log loss = 0.663510454901

Accuracy (with Singular value decomposition Linear_Discriminant_Analysis) = 0.721396250808

Singular value decomposition Logloss (with calibration using isotonic) = 0.751748639402 compare: 0.751748639402


Eigenvalue Decomposition Result
Class 0 log loss = 2.20821019279
Class 1 log loss = 0.594293170288
Class 2 log loss = 1.30102777344
Class 3 log loss = 2.4152429207
Class 4 log loss = 0.286584390889
Class 5 log loss = 0.313558835926
Class 6 log loss = 1.58741023046
Class 7 log loss = 0.528356206075
Class 8 log loss = 0.709104746065

Accuracy (with Eigenvalue decomposition Linear_Discriminant_Analysis) = 0.71800258565

 Eigenvalue decomposition Logloss (with calibration using isotonic) = 0.784122037555 compare: 0.784122037555


'''




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



def Draw(X_r):

	colors=['blue','green','red','cyan','magenta','yellow','black','orange','purple']
	index= [0,1,2,3,4,5,6,7,8]
	targetName=['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9']
	lw = 0.1
	for color,i,target_name in zip(colors,index,targetName):
		plt.scatter(X_r[y==i,0],X_r[y==i,1],color=color,alpha=0.8,lw=lw,label=target_name,marker='*')
	plt.legend(loc='best', shadow=False, scatterpoints=1)
	plt.title('LDA Graph')
	plt.show()
        




#===================Singular value decomposition part=================

def SingularValue_LDA():
    
	clf=LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
	              solver='svd', store_covariance=False, tol=0.0001)
    
	clf_CV= CalibratedClassifierCV(clf, method='isotonic', cv=5)
	clf_fit_CV = clf_CV.fit(Xtrain, ytrain)
	
	#======================================
        clf_fit = clf.fit(Xtrain, ytrain)
	clf_predict= clf_fit_CV.predict(Xtest)
	
	correct=0
	for i in range(len(clf_predict)):
	    if clf_predict[i]==ytest[i]:
		correct += 1
	
	
	#loglossResult=log_loss(ytest,y_pred)
    
	y_pred_CV= clf_fit_CV.predict_proba(Xtest)
	loglossResult = log_loss(ytest, y_pred_CV)
    
	loglossResult_withIsotonic, logloss_list=log_loss_implement(ytest, y_pred_CV)
		
    
    
	print '\nAccuracy (with Singular value decomposition Linear_Discriminant_Analysis) = ' + str(float(correct)/len(ytest))
	
	#print '\nThe Singular value decomposition log loss = ' + str(loglossResult)
    
	print '\nSingular value decomposition Logloss (with calibration using isotonic) = ' + str(loglossResult_withIsotonic) + ' compare:', str(loglossResult)
    
	
	X_r= clf_fit.transform(Xtrain)
	write_pred_prob(y_pred_CV)
	Draw(X_r)
    

#================================================================



   
#=======================Eigenvalue decomposition part================= 
def eigen_LDA():
    
	clf=LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
	              solver='eigen', store_covariance=False, tol=0.0001)
    
	clf_CV= CalibratedClassifierCV(clf, method='isotonic', cv=5)
	clf_fit_CV = clf_CV.fit(Xtrain, ytrain)
	
	#======================================
        clf_fit = clf.fit(Xtrain, ytrain)
	clf_predict= clf_fit_CV.predict(Xtest)
	
	correct=0
	for i in range(len(clf_predict)):
	    if clf_predict[i]==ytest[i]:
		correct += 1
	
	
	#loglossResult=log_loss(ytest,y_pred)
    
	y_pred_CV= clf_fit_CV.predict_proba(Xtest)
	loglossResult = log_loss(ytest, y_pred_CV)
    
	loglossResult_withIsotonic, logloss_list=log_loss_implement(ytest, y_pred_CV)
		
    
    
	print '\nAccuracy (with Eigenvalue decomposition Linear_Discriminant_Analysis) = ' + str(float(correct)/len(ytest))
	
	#print '\nThe Singular value decomposition log loss = ' + str(loglossResult)
    
	print '\n Eigenvalue decomposition Logloss (with calibration using isotonic) = ' + str(loglossResult_withIsotonic) + ' compare:', str(loglossResult)
    
	
	X_r= clf_fit.transform(Xtrain)
	write_pred_prob(y_pred_CV)
	Draw(X_r)
    
SingularValue_LDA()

#eigen_LDA()


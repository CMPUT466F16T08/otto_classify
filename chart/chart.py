import pandas as pd
import numpy as np

filelist=['prediction_prob_cnn.csv','prediction_prob_LDA.csv','prediction_prob_nn.csv','prediction_prob_random_forest.csv','prediction_prob_svm.csv','prediction_prob_xgboost.csv']
namelist=['CNN','LDA','NN','forest','SVM','xgboost']
l=len(namelist)

output=pd.read_csv('../test_set.csv')
tre=output['target']
output=output['id']
output=pd.DataFrame(output,columns=['id'])

for i in range(0,l):
  col=pd.read_csv(filelist[i])
  col=col.values
  col=col[:,1:]
  col=np.argmax(col,axis=1)
  output[namelist[i]]=col
output['true class']=tre

output.to_csv('chart.csv',index=False)


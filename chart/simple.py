import pandas as pd
import numpy as np

output=pd.read_csv('chart.csv')
output=output.values

for i in range(1,output.shape[1]-1):
  output[:,i]=output[:,i]==output[:,-1]

output=pd.DataFrame(output,columns=['id','CNN','KNN','LDA','NN','forest','SVM','xgboost','true'])
output.to_csv('simple.csv',index=False)


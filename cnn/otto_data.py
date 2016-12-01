import pandas as pd
import numpy as np

class process:
  def read_data_sets(self):
    self.x = pd.read_csv('../train_set.csv')
    self.x=self.x.values
    ylabel=self.x[:,-1]
    self.x=self.x[:,1:-1]
    self.y=np.zeros((self.x.shape[0],9))
    self.y[np.arange(self.x.shape[0]),ylabel]=1
    self.count=0
    
  def next_batch(self,n):
    op=self.count
    if(op>self.x.shape[0]):
      count=0
      op=0
    ed=np.minimum(op+n,self.x.shape[0])
    batchx=self.x[op:ed,:]
    batchy=self.y[op:ed,:]
    batch=[batchx,batchy]
    return batch
    
  def testset(self):
    tx=pd.read_csv('../test_set.csv')
    tx=tx.values
    tylabel=tx[:,-1]
    ids=tx[:,0]
    tx=tx[:,1:-1]
    ty=np.zeros((tx.shape[0],9))
    ty[np.arange(tx.shape[0]),tylabel]=1
    testset=[tx,ty,ids]
    return testset

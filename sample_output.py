import pandas as pd
import numpy as np

X = pd.read_csv('test_set.csv')

ids=X.id.values
l=ids.size
rand=np.full((l),1.0/9)

sout=pd.DataFrame.from_items([('id',ids),('Class_1',rand),('Class_2',rand),('Class_3',rand),('Class_4',rand),('Class_5',rand),('Class_6',rand),('Class_7',rand),('Class_8',rand),('Class_9',rand),])
sout.to_csv('sample_output.csv',index=False)


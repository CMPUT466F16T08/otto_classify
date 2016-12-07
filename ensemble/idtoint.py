import pandas as pd
import numpy as np

rx = pd.read_csv('myresult.csv')
rx=rx.values
rid=rx[:,0]

rid=rid.astype(int)

re=pd.DataFrame(rx,columns=['id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']);
re['id']=rid
re.to_csv('myresult.csv',index=False)

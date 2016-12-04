# -*- coding: utf-8 -*-

import numpy as np
from sklearn.manifold import TSNE



import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

train = pd.read_csv('train_set.csv')

train = train[0:15000][:]

train = train.drop('id', axis=1)
y = train.target.values

X = train.drop('target', axis=1)

model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
X_r=model.fit_transform(X) 

plt.figure()
colors = ['green','navy','pink','darkorange','gray','red','purple','turquoise','yellow']
lw = 1
target_names=sorted(list(set(train.target.values)))

for color, i,target_name in zip(colors, range(9),target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1],color=color,label=target_name,lw=lw,alpha=1)
plt.legend(loc='best', shadow=True, scatterpoints=10)
plt.title('TSNE of dataset')
plt.show()
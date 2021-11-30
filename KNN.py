# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 00:47:21 2021

@author: Blade
"""

import matplotlib.pyplot as plt
import numpy as np
from pandas import *
import pandas as pd
from sklearn.datasets import  load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.model_selection import KFold
#%% data preprocessing

X_train = np.array(pd.read_csv('./x_train_pca.csv',header = None))
y_train = np.array(pd.read_csv('./y_train.csv',header = None)).ravel()
X_test = np.array(pd.read_csv('./x_test_pca.csv',header = None))
y_test = np.array(pd.read_csv('./y_test.csv',header = None)).ravel()
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# transform label into 1/0
i = 0
for label in y_train:
    if label == 1:
        y_train[i] = 1
        
    elif label == 2:
        y_train[i] = 1
        
    elif label == 3:
        y_train[i] = 1
        
    i = i+1

i = 0    
for label in y_test:
    if label == 1:
        y_test[i] = 1
        
    elif label == 2:
        y_test[i] = 1
        
    elif label == 3:
        y_test[i] = 1
        
    i = i+1

#%% 
def KNNClassifier(X_train, y_train, X_test,k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train) # Fit KNN model

    y_pred = neigh.predict(X_test)
    return y_pred



#%% 
KNN_k = [x for x in range(1,round(len(X_train)**0.5)+1)]
kf=KFold(n_splits=100,random_state=0,shuffle=True)
k_candidate = []
for k in KNN_k:
    score = 0
    for train_index,valid_index in kf.split(X_train):
        y_pred = KNNClassifier(X_train[train_index], y_train[train_index], X_train[valid_index],k)
        score =score + accuracy_score(y_train[valid_index],y_pred)
    avg_score = score/kf.n_splits
    k_candidate.append(avg_score)
    print('\r'"Cross Validation Process:{0}%".format(round(k * 100 / len(KNN_k))), end="",flush=True)
#%%
k_best =k_candidate.index(max(k_candidate))+1
print('\nBest k: '+str(k_best))

#%%

y_pred=KNNClassifier(X_train, y_train, X_test,k_best)
print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
print(classification_report(y_test,y_pred))

#%%

def k_plotter(X_train, y_train, X_test,k_set):
    #score_T = np.array([[0]*2]*(k_set),float)
    score_T=np.array([[0]*2 for i in range(k_set)],float)
    for k in range(1,k_set+1):
        score = accuracy_score(y_test,KNNClassifier(X_train, y_train, X_test,k))
        score_T[k-1,:] = [score, k]
    
    return score_T


score_k = k_plotter(X_train, y_train, X_test,round(len(X_train)**0.5))

plt.figure()
plt.subplot(1, 2, 1) #图一包含1行2列子图，当前画在第一行第一列图上
plt.plot(KNN_k, k_candidate)


plt.subplot(1, 2, 2)#当前画在第一行第2列图上
plt.plot(score_k[:,1],score_k[:,0])


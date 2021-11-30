
"""
Created on Thu Nov 25 02:27:34 2021

@author: Blade
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import pandas as pd
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
    
    
#%% Use SVM from a library

def SVM(X_train,y_train, X_test, gamma):
    model = svm.SVC(C=1, gamma = gamma)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return y_pred
#%%
gamma_list = np.linspace(0.1,1,20)
n_splits = 10
kf=KFold(n_splits=n_splits,random_state=0,shuffle=True)
gamma_candidate = []
i = 0
for gamma in gamma_list:
    
    score = 0
    for train_index,valid_index in kf.split(X_train):
        y_pred = SVM(X_train[train_index], y_train[train_index], X_train[valid_index],gamma)
        score =score + accuracy_score(y_train[valid_index],y_pred)
        i = i+1
        print('\r'"Cross Validation Process:{0}%".format(round(i *100 / (len(gamma_list)*n_splits))), end="",flush=True)
       
      
    avg_score = score/kf.n_splits
    gamma_candidate.append(avg_score)
    
gamma_best = gamma_list[gamma_candidate.index(max(gamma_candidate))]
print('\nBest Gamma: '+str(gamma_best))

#%%
plt.plot(gamma_list,gamma_candidate)

# Scikit learn library results
y_pred=SVM(X_train,y_train, X_test,gamma_best)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))




















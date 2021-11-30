# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 02:14:28 2021

@author: Blade
"""

import matplotlib.pyplot as plt
import numpy as np
from pandas import *
import pandas as pd
from sklearn.datasets import  load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

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
    

#%%  Complete the function boostingClassifierML by using Boosting classifier built-in function.
def boostingClassifierML(X_train, y_train, X_test, k):
    
    boostmodel = AdaBoostClassifier(n_estimators = k)
    boostmodel.fit(X_train, y_train, sample_weight = None)
    
    for index in range(0, 3):
        plt.figure()
        fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=800)
        tree.plot_tree(boostmodel.estimators_[index],class_names=['No Tumor', 'Have Tumor'], filled = True, rounded=True,proportion = False);
    
    y_pred = boostmodel.predict(X_test)
    
    return y_pred 

k = 300
y_pred = boostingClassifierML(X_train,y_train,X_test, k)
# score = metrics.accuracy_score(y_test,y_pred)

# print('Adaboost performance is: {}'.format(round(score,3))) 
print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
print(classification_report(y_test,y_pred))





    
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 05:00:30 2021

@author: Blade
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patches
from scipy.special import expit
import itertools
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
#%% data preprocessing

x_train = np.array(pd.read_csv('./x_train_lda.csv',header = None))
y_train = np.array(pd.read_csv('./y_train.csv',header = None))
x_test = np.array(pd.read_csv('./x_test_lda.csv',header = None))
y_test = np.array(pd.read_csv('./y_test.csv',header = None))
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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


#%% Use logistic regression from a library

# sklearn functions implementation
def logRegrPredict(x_train, y_train,xtest ):
    # Build Logistic Regression Model
    logreg = LogisticRegression(solver='lbfgs')
    # Train the model using the training sets
    logreg.fit(x_train, y_train)
    y_pred= logreg.predict(xtest)
    #print('Accuracy on test set: {:.2f}'.format(logreg.score(x_test, y_test)))

    return y_pred, logreg.coef_ ,logreg.intercept_
    
y_pred, coef, intercept = logRegrPredict(x_train, y_train,x_test)

print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred)))
print(classification_report(y_test,y_pred))#text report showing the main classification metrics
#%%
# x_set,y_set=x_train,y_train
# X1,X2=np. meshgrid(np. arange(start=x_set[:,0].min()-1, stop=x_set[:, 0].max()+1, step=0.01),np. arange(start=x_set[:,1].min()-1, stop=x_set[:,1].max()+1, step=0.01))
# plt.contourf(X1, X2,logRegrPredict(x_set, y_set,[]) ,alpha = 0.75, cmap = ListedColormap(('red', 'green')))
fig = plt.figure()
plt.rcParams['savefig.dpi'] = 2000       # 图片像素
plt.rcParams['figure.dpi'] = 2000        # 分辨率
ax = fig.add_subplot(111, projection='3d') 

#Transfer to 1/0 label
i = 0
for label in y_pred:
    
    if label == 0:
        ax.scatter(x_test[i,0], x_test[i,1], x_test[i,2], c='#4d3333', s=1, alpha=1)
    
    elif label == 1:
        ax.scatter(x_test[i,0], x_test[i,1], x_test[i,2], c='#3333cc', s=1, alpha=1)
        
   
    elif label == 2:
        ax.scatter(x_test[i,0], x_test[i,1], x_test[i,2], c='#ff1493', s=12, alpha=1)
        
    elif label == 3:
        ax.scatter(x_test[i,0], x_test[i,1], x_test[i,2], c='green', s=12, alpha=1)
        
    i = i+1        
    

    
#%%

fig = plt.figure()
plt.rcParams['savefig.dpi'] = 2000       # 图片像素
plt.rcParams['figure.dpi'] = 2000        # 分辨率

i = 0
for label in y_test:
    
    if label == 0:
        plt.scatter(x_test[i,1], x_test[i,2], c='#4d3333', s=5, alpha=1)
    
    elif label == 1:
        plt.scatter(x_test[i,1], x_test[i,2], c='#3333cc', s=5, alpha=1)
        
   
    elif label == 2:
        plt.scatter(x_test[i,1], x_test[i,2], c='#ff1493', s=5, alpha=1)
        
    elif label == 3:
        plt.scatter(x_test[i,1], x_test[i,2], c='green', s=5, alpha=1)
        
    i = i+1        


  
def x2(x1):
    return (coef[0,1] * x1 - intercept) / coef[0,2]


x1_plot = np.linspace(0, 1, 1000)
x2_plot = x2(x1_plot)


plt.plot(x1_plot, x2_plot,c='red')










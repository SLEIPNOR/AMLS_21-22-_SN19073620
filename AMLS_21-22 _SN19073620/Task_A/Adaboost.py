# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 02:14:28 2021

@author: Z.Han
"""
""" Hello, I'm so happy you can use my code. I have prepared all preprocessed data so you 
can use this code to reproduce the hyperparameter selection and training process directly. 
Just run each code block sequently by press "ctrl + enter". Note here you can adjust max_depth,
n_estimators, and learning_rate.
If you only want to test the final model with the best hyperparameter, just run 3rd code 
block "formal training and testing" """
#%% import module & def func
from PIL import Image
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score


def boostingClassifierML(X_train, y_train, X_test, k):

    estimatorCart = DecisionTreeClassifier(max_depth=1) # change hyper-parameter max_depth
    boostmodel = AdaBoostClassifier(base_estimator = estimatorCart
                                    ,n_estimators = k,learning_rate=1) # change n_estimators, learning rate
    boostmodel.fit(X_train, y_train, sample_weight = None) # initialize sample weight at first

    # plot a decision tree
    # for index in range(0, 2):
    #     plt.figure(figsize=(12, 5))
    #     tree.plot_tree(boostmodel.estimators_[index],class_names=['No Tumor', 'Have Tumor'], filled = True, rounded=True,proportion = False);
    #     plt.tight_layout()
    #     plt.show()

    y_pred = boostmodel.predict(X_test) # val_data predict
    y_pred_T = boostmodel.predict(X_train) # train_data predict
    
    return y_pred, y_pred_T

#%% five-fold cross-validation

""" code below implement AdaBoost with different n_estimators in a fixed range in five-fold 
cross-validation and also plot a 3D figure"""

# set a range of n_estimators to test
k_candidate = np.linspace(50, 300, 51).tolist()
score_avg = list()
# test start
for k in k_candidate:
    print("k candidate {}".format(k))
    score = list()

    # 5-fold cross-validation
    for i in range(5):
        print("fold {}".format(i))

        # load preprocessing data in i-fold
        X_train = np.array(pd.read_csv('./x_train_pca_{}.csv'.format(i),header = None))
        y_train = np.array(pd.read_csv('./y_train_{}.csv'.format(i),header = None)).ravel()
        X_test = np.array(pd.read_csv('./x_test_pca_{}.csv'.format(i),header = None))
        y_test = np.array(pd.read_csv('./y_test_{}.csv'.format(i),header = None)).ravel()

        # normalize
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # training & prediction
        y_pred,_ = boostingClassifierML(X_train,y_train,X_test,int(k))
        score.append(accuracy_score(y_test,y_pred))  # collect acc score

        print('Accuracy: '+str(accuracy_score(y_test,y_pred)))
    score_avg.append(np.mean(score))   # collect avg acc score
    print('avg Accuracy: ' + str(score_avg[k_candidate.index(k)]))

# np.savetxt('ada_score_lr=1.5.csv', score_avg, delimiter = ',')
# score_avg = pd.read_csv('ada_score.csv',header=None)
# k_candidate = np.linspace(50, 300, 51).tolist()

plt.figure()
plt.plot(k_candidate,score_avg)
plt.xlabel('num of classifiers')
plt.ylabel('avg acc')
# if u want to save img
# plt.savefig('ada_score.png',dpi=600)
plt.show()

""" The code below is for learning curve plotting (best model) in five-fold cross-validation, which study the relationship between 
the number of training samples and train_acc, val_acc, just uncomment and then you can use 
it to plot ! """

# k_list = [400,800,1200,1600,2000,2400] # define the size of the training data
# score_avg_val = []
# score_avg_train = []
# for k in k_list:
#     score_T = []
#     score =[]
#     for i in range(5):
#         print("fold {}".format(i))
#
#         X_train = np.array(pd.read_csv('./x_train_pca_{}.csv'.format(i), header=None))
#         y_train = np.array(pd.read_csv('./y_train_{}.csv'.format(i), header=None)).ravel()
#         X_test = np.array(pd.read_csv('./x_test_pca_{}.csv'.format(i), header=None))
#         y_test = np.array(pd.read_csv('./y_test_{}.csv'.format(i), header=None)).ravel()
#         scaler = MinMaxScaler()
#         X_train = scaler.fit_transform(X_train[:k,:])
#         X_test = scaler.transform(X_test)
#
#         y_pred, y_pred_T = boostingClassifierML(X_train,y_train[:k],X_test,165)
#
#         score_T.append(accuracy_score(y_train[:k], y_pred_T)) # collect training data acc score
#         score.append(accuracy_score(y_test, y_pred))  # collect test data acc score
#         print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
#     score_avg_val.append(np.mean(score))  # collect test data avg acc score
#     score_avg_train.append(np.mean(score_T))  # collect training data avg acc score
#
# plt.plot(k_list, score_avg_val, label='val_acc')
# plt.plot(k_list, score_avg_train, label='train_acc')
# plt.xlabel('num of samples')
# plt.ylabel('avg acc ')
# plt.legend(loc=4)
# plt.savefig('ada_best.png',dpi=600)
# plt.show()

#%% formal training and testing

# load preprocessed training & test data
X_train = np.array(pd.read_csv('./x_train_pca_formal.csv',header = None))
y_train = np.array(pd.read_csv('./y_train_formal.csv',header = None)).ravel()
X_test = np.array(pd.read_csv('./x_test_pca_formal.csv',header = None))
y_test = np.array(pd.read_csv('./y_test_formal.csv',header = None)).ravel()

# training & prediction
y_pred, y_pred_T = boostingClassifierML(X_train,y_train,X_test,165)
# accuracy_score(y_test, y_pred)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
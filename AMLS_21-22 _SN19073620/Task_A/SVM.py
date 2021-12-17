
"""
Created on Thu Nov 25 02:27:34 2021

"""
""" Hello, I'm so happy you can use my code. I have prepared all preprocessed data so you 
can use this code to reproduce the grid search and training process directly. 
Just run each code block sequently by press "ctrl + enter".
 If you only want to test the final model with the best C and gamma, just run 3rd code 
 block "formal training and testing" """
#%% import module & def func

import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

# define SVM func
def SVM(X_train,y_train, X_test, gamma,C):
    model = svm.SVC(C=C, gamma = gamma,kernel='rbf')
    # model = svm.SVC(C=1, kernel='linear') # test linear

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)  # test data prediction
    y_pred_T = model.predict(X_train) # training data prediction

    return y_pred, y_pred_T


# %% five-fold cross-validation
""" code below implement a gird search for C and gamma in five-fold 
cross-validation and also plot a 3D figure"""

# grid search range
C_list= [1e-2,1e-1,1e0,1e1,2e1,3e1,4e1,5e1] #  Regularization
gamma_list = [1e-3,1e-2,1e-1,1e0,1e1,2e1,3e1,4e1,5e1] # gamma for kernel func

# grid search start
score_avg = list()
for C in C_list:
    print("C candidate {}".format(C))
    for gamma in gamma_list:

        print("gamma candidate {}".format(gamma))
        score = list()

        # 5-fold cross-validation
        for i in range(5):
            print("fold {}".format(i))

            # load preprocessing data in i-fold
            X_train = np.array(pd.read_csv('./x_train_pca_{}.csv'.format(i),
                                           header=None))
            y_train = np.array(pd.read_csv('./y_train_{}.csv'.format(i),
                                           header=None)).ravel()
            X_test = np.array(pd.read_csv('./x_test_pca_{}.csv'.format(i),
                                          header=None))
            y_test = np.array(pd.read_csv('./y_test_{}.csv'.format(i),
                                          header=None)).ravel()

            # normalize
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # training & prediction
            y_pred, _ = SVM(X_train,y_train, X_test,gamma,C)

            score.append(accuracy_score(y_test, y_pred)) # collect acc score

            print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
        score_avg.append(np.mean(score))  # collect avg acc score
        print('avg Accuracy: ' + str(score_avg[gamma_list.index(gamma)]))

    # print('avg Accuracy: ' + str(score_avg[C_list.index(C)]))
# np.savetxt('grid_search.csv', score_avg, delimiter = ',')
# score_avg = np.array(pd.read_csv('grid_search.csv',header=None))

fig = plt.figure()  # 3D axis plot start !
ax3 = plt.axes(projection='3d')
# construct mesh
# log scale change
xx = np.array([np.log10(1e-3),np.log10(1e-2),np.log10(1e-1),np.log10(1e0),
               np.log10(1e1),np.log10(2e1),np.log10(3e1),np.log10(4e1),np.log10(5e1)])
yy = np.array([np.log10(1e-2),np.log10(1e-1),np.log10(1e0),np.log10(1e1),
               np.log10(2e1),np.log10(3e1),np.log10(4e1),np.log10(5e1)])
X, Y = np.meshgrid(xx, yy)


ax3.plot_surface(X,Y,np.reshape(score_avg,[8,9]),cmap='cubehelix')
ax3.set_ylabel('log(C)')
ax3.set_xlabel('log(gamma)')
ax3.set_zlabel('avg val_acc')

# if u want to save img
# plt.savefig('SVM_gamma_C.png',dpi=600)

plt.show()

""" The code below is for learning curve plotting (best model) in five-fold cross-validation,
which study the relationship between the number of training samples and train_acc, val_acc, 
just uncomment and then you can use it to plot ! """

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
#         y_pred, y_pred_T = SVM(X_train, y_train[:k], X_test, 1e-2, 4e1)
#
#         score_T.append(accuracy_score(y_train[:k], y_pred_T))   # collect training data acc score
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
# # plt.savefig('SVM_best.png',dpi=600)
# plt.show()

#%% formal training and testing

# load preprocessed training & test data
X_train = np.array(pd.read_csv('./x_train_pca_formal.csv',header = None))
y_train = np.array(pd.read_csv('./y_train_formal.csv',header = None)).ravel()
X_test = np.array(pd.read_csv('./x_test_pca_formal.csv',header = None))
y_test = np.array(pd.read_csv('./y_test_formal.csv',header = None)).ravel()

# training & prediction
y_pred, y_pred_T = SVM(X_train, y_train, X_test, 1e-2, 4e1)
# accuracy_score(y_test, y_pred)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

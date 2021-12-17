"""This plotter file is used to analysis the test results, all figure are collected in
Assignment Report"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% plot ada with different hyperparameters

a1 = pd.read_csv('ada_score_lr=0.2.csv',header=None)
a2 = pd.read_csv('ada_score_lr=0.6.csv',header=None)
a3 = pd.read_csv('ada_score_lr=1.csv',header=None)
a4 = pd.read_csv('ada_score_lr=1.5.csv',header=None)
a5 = pd.read_csv('ada_score_lr=1_dep=2.csv',header=None)
k_candidate = np.linspace(50, 300, 51).tolist()

plt.figure(figsize=(10, 5))
# plotting loss vs lr
plt.subplot(1, 2, 1)
plt.plot(k_candidate,a1,label='lr=0.2 ')
plt.plot(k_candidate,a2,label='lr=0.6 ')
plt.plot(k_candidate,a3,label='lr=1')
plt.plot(k_candidate,a4,label='lr=1.5')


plt.xlabel('n_estimators ')
plt.ylabel('avg acc')
plt.legend(loc=4)

plt.subplot(1, 2, 2)
plt.plot(k_candidate,a3,label='max_depth = 1, lr = 1 ')
plt.plot(k_candidate,a5,label='max_depth = 2, lr = 1 ')
plt.legend(loc=4)
plt.xlabel('n_estimators ')
plt.ylabel('avg acc')
plt.savefig('ada.pdf',dpi=600)
plt.show()

#%% plot confusion matrix indicators


name_list = ('precision_0', 'recall_0', 'precision_1', 'recall_1')
num_list = [0.97, 0.76, 0.95, 0.99]
num_list1  = [1.00 , 0.49, 0.90 , 1.00]

bar_width = 0.3
index_ada = np.arange(len(name_list))
index_svm = index_ada + bar_width


plt.bar(index_ada, height=num_list, width=bar_width, color='b', label='AdaBoost')
plt.bar(index_svm, height=num_list1, width=bar_width, color='g', label='SVM')
plt.ylim(0,1.3)
plt.legend(loc = 1)
plt.xticks(index_ada + bar_width/2, name_list)
plt.savefig('nondeep_result.png',dpi=600)
plt.show()





















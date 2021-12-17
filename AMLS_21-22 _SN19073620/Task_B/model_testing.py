
""" This main file is used to test our model's performance"""


#%% import module & def func

import tensorflow as tf
import ipykernel
from tensorflow.keras import layers, models, regularizers, Model, callbacks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import data_reading as dr
import numpy as np
import gc
import random
import data_augmentation as aug
import basic_experts_model as bem
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import glob
import basic_experts_model as bem

# load and down sampling
def loading_test_data():
    test_images, test_labels = dr.tumor_4class_test()
    test_images = aug.down_sampling(test_images, size=128)

    return test_images,test_labels


# ensemble prediction
def ensemble_pred(experts, mode):

    """experts must be a list whatever the size it is, when mode = 'avg',
    code below is for probability merging, when mode = 'vote', code below
    is for majority vote """

    pred = np.array([model.predict(test_images) for model in experts])

    if mode == 'avg':
        mean = np.mean(pred, axis=0)
        result = np.argmax(mean, axis=1)

    elif mode == 'vote':

        pred = np.argmax(pred, axis=2)
        result = list()
        for i in range(len(test_labels)):
            result.append(random.sample({k: v for k, v in Counter(pred[:, i]).items()
                                         if v == max(Counter(pred[:, i]).values())}.keys(), 1)[0])

    return result

#%% loading test data

test_images,test_labels = loading_test_data()
test_images = test_images/255

#%% load ten models (you can change model A or model B)
experts = list()
for i in range(10):

    model = bem.ResNet_expert(test_images)
    model.load_weights('./checkpoints_modelB/my_checkpoint{}'.format(i+1))
    experts.append(model)
#%% model evaluation
# print each experts accuracy
# for i in range(10):
#
#     print(accuracy_score(test_labels,np.argmax(experts[i].predict(test_images), axis=1)))

""" Here, you could change vote or avg to make ensemble prediction"""
print(confusion_matrix(test_labels,ensemble_pred(experts,mode='vote')))
print(accuracy_score(test_labels, ensemble_pred(experts,mode='vote')))
print(classification_report(test_labels,ensemble_pred(experts,mode='vote')))

""" Code below is used to plot the relationship between the number of experts and
prediction accuracy from 1 to 10 by using avg and vote, if you want to plot,
just uncomment the code below"""

# num = 10
# # avg prediction
# ensemble_score =list()
# for i in range(num):
#
#     ensemble_score.append(accuracy_score(test_labels
#                                          , ensemble_pred(experts[:i+1],mode='avg')))
#
# ensemble_score_avg = ensemble_score
#
#
# # vote prediction
# ensemble_score =list()
# for i in range(num):
#
#     ensemble_score.append(accuracy_score(test_labels
#                                          , ensemble_pred(experts[:i+1],mode='vote')))
#
# ensemble_score_vote = ensemble_score
#
# print(classification_report(test_labels,ensemble_pred(experts ,mode='avg')))
#
# #plot Num vs Acc
#
# plt.plot( [i for i in range(1,num+1)],ensemble_score_avg)
# plt.plot( [i for i in range(1,num+1)],ensemble_score_vote)
# my_x_ticks = np.arange(1, num+1, 1)
# plt.xticks(my_x_ticks)
# plt.legend(['average probability', 'majority vote'],loc = 1)
# plt.xlabel("number of experts in ensemble model")
# plt.ylabel("accuracy for ensemble model")
# # plt.grid()
# plt.show()


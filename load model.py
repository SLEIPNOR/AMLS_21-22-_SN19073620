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
from sklearn.metrics import classification_report

def loading_test_data():
    images, labels = dr.tumor_4class()
    images = aug.down_sampling(images, size=64)
    train_images, test_images, train_labels, test_labels \
        = train_test_split(images, labels, test_size=0.25, random_state=4)

    return test_images,test_labels



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


#%% loading model
experts = list()
num = 20
for i in range(num):
    experts.append(models.load_model('ensemble(ResNet_no_aug)'+'/expert_{}/'.format(i)))
    print('\r'"loading models:{0}%".format(round((i + 1) * 100 / num))
          , end="", flush=True)
#%%

test_images,test_labels = loading_test_data()

#%%

# avg prediction
ensemble_score =list()
for i in range(num):

    ensemble_score.append(accuracy_score(test_labels
                                         , ensemble_pred(experts[:i+1],mode='avg')))

ensemble_score_avg = ensemble_score


# vote prediction
ensemble_score =list()
for i in range(num):

    ensemble_score.append(accuracy_score(test_labels
                                         , ensemble_pred(experts[:i+1],mode='vote')))

ensemble_score_vote = ensemble_score

print(classification_report(test_labels,ensemble_pred(experts ,mode='avg')))

#%% plot Num vs Acc

plt.plot( [i for i in range(1,num+1)],ensemble_score_avg)
plt.plot( [i for i in range(1,num+1)],ensemble_score_vote)
my_x_ticks = np.arange(1, num+1, 1)
plt.xticks(my_x_ticks)
plt.legend(['Average probability', 'Majority vote'],loc = 4)
plt.xlabel("Number of experts in ensemble model")
plt.ylabel("Accuracy for ensemble model")
plt.grid()
plt.show()

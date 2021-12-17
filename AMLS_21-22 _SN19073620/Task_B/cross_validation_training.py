
"""This module is used for hyper-parameter adjustment and also give a general performance
evaluation to our CNN model by using ten-fold cross-validation, and also generate an ensemble
learning which is consist of ten models from ten-fold cross-validation"""
#%% import module & def func

import tensorflow as tf
import ipykernel
from tensorflow.keras import  models, callbacks
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
from sklearn.model_selection import KFold
from keras.callbacks import LearningRateScheduler
import os

# test learning rate
def lr_scheduler(epoch,callbacks):

    add = 1e-5

    if epoch == 0:
        return callbacks

    return callbacks + add * pow(10, np.floor((epoch-1) / 9))

# decay learning rate when training
def lr_decay(epoch,lr):
    decay = 10
    ms= [20,30,40,50,60]

    if epoch == ms[0]:
        return lr/decay

    elif epoch == ms[1]:
        return lr/decay

    elif epoch == ms[2]:
        return lr/decay

    elif epoch == ms[3]:
        return lr/decay

    elif epoch == ms[4]:
        return lr/decay

    return lr




# load dataset
def load_ds(size):
    images, labels = dr.tumor_4class()

    images = aug.down_sampling(images, size=size)

    return images, labels

# training model
def model_training(train_images, test_images, train_labels, test_labels, lr):

    # building model
    opt = tf.keras.optimizers.Adam(lr)


    model.compile(optimizer = opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    early_stopping = callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0,
                                                   patience=40, verbose=0, mode='max',
                                                   baseline=None, restore_best_weights=True)




    history = model.fit(train_images, train_labels, epochs=200,
                        validation_data=(test_images, test_labels), callbacks=[early_stopping
            ,LearningRateScheduler(lr_decay, verbose=1)])



    return model, history

#ensemble prediction
def ensemble_pred(experts,test_images,test_labels,mode):

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

#%% loading data & down sampling

images, labels = load_ds(128)

#%% finding best learning rate and plot

images = images / 255.0
lr = [1e-5]
model = bem.ResNet_expert(images)
callbacks = [LearningRateScheduler(lr_scheduler, verbose=1)]
model.compile(optimizer= tf.keras.optimizers.Adam(lr[0]),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False))
model.summary()
history = model.fit(images, labels, callbacks=callbacks, epochs=46, verbose=1)

# learning rate range
lr = [1e-5]
for i in range(45):
    lr.append(lr[i] + lr[0] * pow(10, np.floor((i) / 9)))

# plotting loss vs lr
plt.plot(lr,history.history['loss'])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('learning rate (log scale)')
plt.ylabel('loss')
plt.show()

# plotting epochs vs lr
plt.plot(np.linspace(1,46,46),lr)
plt.yscale('log')
plt.xlabel('epochs')
plt.ylabel('learning rate (log scale)')
plt.show()

#save result
# np.savetxt('dropout_only.csv', history.history['loss'], delimiter = ',')





""" hyper_para punishment elements for regularization are also tested in this whole 
data_set to show that with the regularization, the loss function could be 
difficult to be optimized"""

#%% ten-fold cross-validation & ensemble learning

ten_f=KFold(n_splits=10,random_state=0,shuffle=True)

""" Test if dropout is needed, and give a general performance estimation to our CNN model"""

scores, experts = list(), list()

time = 0
# 10-fold cross-validation
for train_index,valid_index in ten_f.split(images):

    print('\ncross validation {}'.format(time + 1))
    train_images, train_labels = images[train_index], labels[train_index]
    valid_images, valid_labels = images[valid_index], labels[valid_index]

    #data augmentation time
    Aug_Times = 1
    for i in range(Aug_Times):
        train_images_aug, train_labels_aug = aug.data_aug(train_images, train_labels,
                                              crop_min_max=[128, 128], export=False)

        train_images = np.concatenate((train_images, train_images_aug), axis=0)
        train_labels = np.concatenate((train_labels, train_labels_aug), axis=0)

        del train_images_aug, train_labels_aug
        gc.collect()

    train_images, valid_images = train_images / 255.0, valid_images / 255.0

    model = bem.ResNet_expert(train_images)

    # load model for transfer learning
    # model.load_weights('./checkpoints_full_aug/my_checkpoint{}'.format(time+1))

    lr = 1e-3
    model, history = model_training(train_images, valid_images,
                                    train_labels, valid_labels, lr)
    print('>%.3f' % max(history.history['val_accuracy']))
    scores.append(max(history.history['val_accuracy']))
    experts.append(model)
    # save model
    # model.save_weights('./checkpoints/my_checkpoint{}'.format(time+1))
    time += 1

print('Estimated Accuracy %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
# save fig
# plt.savefig('accu_middle_dropout.png',dpi=600)
plt.show()

# save results
# np.savetxt('modelB_aug_acc.csv', [history.history['accuracy'],history.history['val_accuracy']], delimiter = ',')
# np.savetxt('modelB_aug_loss.csv', [history.history['loss'],history.history['val_loss']], delimiter = ',')
# np.savetxt('modelB_aug_scores.csv', scores, delimiter = ',')













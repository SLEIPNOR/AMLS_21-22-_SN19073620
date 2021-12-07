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

def data_split(Aug_Times):
    images, labels = dr.tumor_4class()
    images = aug.down_sampling(images,size = 64)

    train_images, test_images, train_labels, test_labels \
        = train_test_split(images, labels, test_size=0.25, random_state=4)

    for i in range(Aug_Times):
        train_images_aug, train_labels_aug = aug.data_aug(train_images, train_labels,
                                                          crop_min_max=[32, 64], export=False)

        train_images = np.concatenate((train_images, train_images_aug), axis=0)
        train_labels = np.concatenate((train_labels, train_labels_aug), axis=0)

        del train_images_aug, train_labels_aug
        gc.collect()

    return train_images, test_images, train_labels, test_labels

def bagging(train_images, train_labels, n_samples):
    train_images_b = []
    train_labels_b = []
    index = [i for i in range(len(train_labels))]
    slice = random.sample(index, n_samples)
    for i in slice:
        train_images_b.append(train_images[i])
        train_labels_b.append(train_labels[i])

    return np.array(train_images_b), np.array(train_labels_b)


def model_training(train_images, test_images, train_labels, test_labels):

    # building model
    model = bem.ResNet_expert(train_images)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    early_stopping = callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0,
                                                   patience=40, verbose=0, mode='max',
                                                   baseline=None, restore_best_weights=True)



    history = model.fit(train_images, train_labels, epochs=200,
                        validation_data=(test_images, test_labels), callbacks=[early_stopping])



    return model, max(history.history['val_accuracy'])

def ensemble_pred(experts,test_images,mode):

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

#%%
train_images, test_images, train_labels, test_labels = data_split(Aug_Times=0)



#%%
experts_num  = 5
scores, experts = list(), list()
for i in range(experts_num):
    train_images_sub, train_labels_sub = bagging(train_images, train_labels, 2250)
    print('\nexpert model {}'.format(i+1))
    model, test_acc = model_training(train_images_sub, test_images, train_labels_sub, test_labels)
    print('>%.3f' % test_acc)
    scores.append(test_acc)
    experts.append(model)
print('Estimated Accuracy %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
#%%

# avg prediction
ensemble_score =list()
for i in range(len(scores)):

    ensemble_score.append(accuracy_score(test_labels
                                         , ensemble_pred(experts[:i+1],test_images,mode='avg')))

ensemble_score_avg = ensemble_score


# vote prediction
ensemble_score =list()
for i in range(len(scores)):

    ensemble_score.append(accuracy_score(test_labels
                                         , ensemble_pred(experts[:i+1],test_images,mode='vote')))

ensemble_score_vote = ensemble_score
#%%
print(classification_report(test_labels,ensemble_pred(experts,test_images ,mode='avg')))
#%% plot Num vs Acc

plt.plot( [i for i in range(1,len(scores)+1)],ensemble_score_avg)
plt.plot( [i for i in range(1,len(scores)+1)],ensemble_score_vote)
my_x_ticks = np.arange(1, len(scores)+1, 1)
plt.xticks(my_x_ticks)
plt.legend(['Average probability', 'Majority vote'],loc = 4)
plt.xlabel("Number of experts in ensemble model")
plt.ylabel("Accuracy for ensemble model")
plt.grid()
plt.show()

#%% model save
for i in range(15,len(scores)):
    experts[i].save('ensemble(ResNet_no_aug)'+'/expert_{}/'.format(i))

#%%

#%%
# experts[1].save('ensemble(ResNet_no_aug)'+'/expert_{}/'.format(1))




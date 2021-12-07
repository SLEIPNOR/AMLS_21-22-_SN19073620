import tensorflow as tf
import ipykernel
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import data_reading as dr
import numpy as np
import gc
import data_augmentation as aug
#%%
images, labels = dr.tumor_4class()
#%%
train_images, test_images, train_labels, test_labels \
    = train_test_split(images, labels, test_size=0.25, random_state=4)
del images, labels
gc.collect()
#%%
Aug_Times = 2
for i in range(Aug_Times):
    train_images_aug, train_labels_aug = aug.data_aug(train_images,train_labels,
                                                      crop_min_max = [100,500],export = False)

    train_images = np.concatenate((train_images, train_images_aug), axis=0)
    train_labels = np.concatenate((train_labels, train_labels_aug), axis=0)

    del train_images_aug, train_labels_aug
    gc.collect()
#%%
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
#%%

# class_names = ['no_tumor', 'meningioma_tumor', 'glioma_tumor', 'pituitary_tumor']
#
# plt.figure(figsize=(10, 10))
#
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i]) # cmap ='gray'
#     # The CIFAR labels happen to be arrays,
#     # which is why you need the extra index
#     plt.xlabel(class_names[train_labels[i][0]])
#
#
# plt.show()

#%%

model = models.Sequential()
model.add(layers.Conv2D(64, (5, 5), input_shape=(train_images.shape[1], train_images.shape[2], train_images.shape[3])))
model.add(layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.Activation('relu'))

model.add(layers.Conv2D(32, (5, 5)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.Activation('relu'))


model.add(layers.Conv2D(16, (5, 5)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.Activation('relu'))














#%%

model.summary()
#%%

model.add(layers.Flatten())
model.add(layers.Dense(400,kernel_regularizer=regularizers.l2(0.001)))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(120,kernel_regularizer=regularizers.l2(0.001))) #kernel_regularizer=regularizers.l2(0.001)
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(84,kernel_regularizer=regularizers.l2(0.001)))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(4))






model.summary()
#%%

opt = tf.keras.optimizers.Adam(learning_rate=0.000001)

model.compile(optimizer= opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=5,
                    validation_data=(test_images, test_labels))


#%%



plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


#%%
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
#%%
predictions = probability_model.predict(test_images)
# predictions = np.where(predictions == np.max(predictions,axis = 1) )
predictions = np.argmax(predictions, axis=1).reshape(len(predictions),1)
# predictions_linear = model.predict(test_images)
#%%
class_names = ['no_tumor', 'meningioma_tumor', 'glioma_tumor', 'pituitary_tumor']
plt.figure(figsize=(15, 15))

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i]) # cmap ='gray'
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    if predictions[i][0] == test_labels[i][0]:
        plt.xlabel('{}, hit'.format(class_names[predictions[i][0]]))
    else:
        plt.xlabel('{}, miss\nReal: {}'.format(class_names[predictions[i][0]],class_names[test_labels[i][0]]))


plt.show()

import tensorflow as tf
import ipykernel
from keras import regularizers
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import data_reading as dr
import numpy as np
import gc
#%%
images, labels = dr.tumor_4class()
#%%
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.25, random_state=0)
del images, labels
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
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(train_images.shape[1], train_images.shape[2], train_images.shape[3])))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(48, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
#%%

model.summary()
#%%

model.add(layers.Flatten())
model.add(layers.Dense(254, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu')) #kernel_regularizer=regularizers.l2(0.001)
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4))
model.summary()
#%%

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=30,
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

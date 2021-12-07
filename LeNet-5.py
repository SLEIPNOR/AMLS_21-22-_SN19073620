import tensorflow as tf
import ipykernel
from tensorflow.keras import layers, models, activations, regularizers
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
# # train_images = tf.image.random_crop(train_images, [512, 512, 1])
#
# # train_images=tf.image.random_flip_left_right(train_images)
# #
# # train_images=tf.image.random_flip_left_right(train_images)
#
# train_images=tf.image.random_flip_up_down(train_images)
#
# # train_images=tf.image.random_brightness(train_images,0.5)
# #
# # train_images=tf.image.random_contrast(train_images,0,1)
#
# plt.imshow(train_images)
#
# plt.show()


#%%
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

#%%

model = models.Sequential()
model.add(layers.Conv2D(64, (5, 5), input_shape=(512, 512, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.Activation('relu'))

model.add(layers.Conv2D(32, (5, 5)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.Activation(activations.relu))

model.add(layers.Conv2D(16, (5, 5)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.Activation(activations.relu))

#%%

model.summary()
#%%

model.add(layers.Flatten())
model.add(layers.Dense(400,kernel_regularizer=regularizers.l2(0.001)))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(120,kernel_regularizer=regularizers.l2(0.001)))
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

# opt = tf.keras.optimizers.Adam(learning_rate=0.00001)


model.compile(optimizer= 'adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=50,
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
predictions = probability_model.predict(test_images)
# predictions = np.where(predictions == np.max(predictions,axis = 1) )
predictions = np.argmax(predictions, axis=1).reshape(len(predictions),1)
# predictions_linear = model.predict(test_images)

class_names = ['no_tumor', 'meningioma_tumor', 'glioma_tumor', 'pituitary_tumor']


plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5,5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i]) # cmap ='gray'

    if predictions[i][0] == test_labels[i][0]:
        plt.xlabel('{}, hit'.format(class_names[predictions[i][0]]))
    else:
        plt.xlabel('{}, miss'.format(class_names[predictions[i][0]]))


plt.show()






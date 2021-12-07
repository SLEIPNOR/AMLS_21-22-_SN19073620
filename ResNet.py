import tensorflow as tf
import ipykernel
from tensorflow.keras import layers, models, regularizers, Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import data_reading as dr
import numpy as np
import gc
import data_augmentation as aug
from ResBlock import ResnetBlock
#%%
images, labels = dr.tumor_4class()
images = aug.down_sampling(images,size = 64)

#%%
train_images, test_images, train_labels, test_labels \
    = train_test_split(images, labels, test_size=0.25, random_state=4)

del images, labels
gc.collect()
#%%
Aug_Times = 1
for i in range(Aug_Times):
    train_images_aug, train_labels_aug = aug.data_aug(train_images,train_labels,
                                                      crop_min_max = [16,32],export = False)

    train_images = np.concatenate((train_images, train_images_aug), axis=0)
    train_labels = np.concatenate((train_labels, train_labels_aug), axis=0)

    del train_images_aug, train_labels_aug
    gc.collect()

#%%
train_images, test_images = train_images / 255.0, test_images / 255.0
#%%
i = 110
class_names = ['no_tumor', 'meningioma_tumor', 'glioma_tumor', 'pituitary_tumor']
plt.imshow(train_images[i],cmap='gray')
plt.xlabel(class_names[train_labels[i][0]])
plt.show()
#%%
# Create an instance of the model
S = train_images.shape
model = models.Sequential()
chl =64
# First 3*3*64 stride = 2 cov layer with pooling
model.add(layers.Conv2D(chl, (3, 3),strides = 2, input_shape= S[1:]))
model.add(layers.MaxPooling2D(pool_size = (2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

# ResNet block 64
model.add(ResnetBlock(3,[0,chl]))

model.add(ResnetBlock(3,[0,chl]))

model.add(ResnetBlock(3,[0,chl]))

model.add(ResnetBlock(3,[0,chl]))

model.add(ResnetBlock(3,[0,chl]))

model.add(layers.AveragePooling2D(pool_size = (2, 2)))

# ResNet block 64
# model.add(ResnetBlock(3,[0,chl]))
#
# model.add(ResnetBlock(3,[0,chl]))
#
# model.add(ResnetBlock(3,[0,chl]))
#
# model.add(ResnetBlock(3,[0,chl]))
#
# model.add(ResnetBlock(3,[0,chl]))
#
# model.add(layers.AveragePooling2D(pool_size = (2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1024))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
#
model.add(layers.Dense(128)) #kernel_regularizer=regularizers.l2(0.001)
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))


model.add(layers.Dense(4,activation='softmax'))

model.summary()
#%%

opt = tf.keras.optimizers.Adam()
# opt =  tf.optimizers.SGD(learning_rate =0.1)
model.compile(optimizer= opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


history = model.fit(train_images, train_labels, epochs=50,
                    validation_data=(test_images, test_labels))

 #%%







#%%



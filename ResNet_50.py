import tensorflow as tf
import ipykernel
from tensorflow.keras import layers, models, activations
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import data_reading as dr
import numpy as np
import gc
# 自动分配显存
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(
    per_process_gpu_memory_fraction=1)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

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

model = tf.keras.applications.resnet50.ResNet50(
    weights=None,
    input_shape=(512, 512, 1), classes=4)
#%%
model.compile(optimizer="Adam",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=20,
                    validation_data=(test_images, test_labels))
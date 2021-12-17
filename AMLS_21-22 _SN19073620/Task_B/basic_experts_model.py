""" Self made module by @ Z.Han"""

""" This module is 24LD model which used in cross_validation_training.py
you can adjust regularization penalty (reg), and you can also add or delete
the dropout layer in FC layers"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from ResBlock import ResnetBlock

def ResNet_expert(train_images):
    trainable = True # fix batchnorm parameter or not
    reg = 0 # regularization
    S = train_images.shape
    model = models.Sequential()
    chl = 64 # channel

    # First 3*3*64 stride = 2 cov layer with max pooling
    model.add(layers.Conv2D(chl, (3, 3), strides=2, input_shape=S[1:]))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization(trainable = trainable)) #batchnorm
    model.add(layers.Activation('relu')) # activation

    # ResNet block 3*3*64
    model.add(ResnetBlock(3, [0, chl]))

    model.add(ResnetBlock(3, [0, chl]))

    model.add(ResnetBlock(3, [0, chl]))

    model.add(ResnetBlock(3, [0, chl]))

    model.add(ResnetBlock(3, [0, chl]))

    # avg pooling
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    # ResNet block 3*3*64
    model.add(ResnetBlock(3, [0, chl]))

    model.add(ResnetBlock(3, [0, chl]))

    model.add(ResnetBlock(3, [0, chl]))

    model.add(ResnetBlock(3, [0, chl]))

    model.add(ResnetBlock(3, [0, chl]))

    # avg pooling
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    # 3 FC layers
    model.add(layers.Flatten())

    model.add(layers.Dropout(0.5)) # drop out

    model.add(layers.Dense(512,kernel_regularizer=regularizers.l2(reg)))
    model.add(tf.keras.layers.BatchNormalization(trainable = trainable)) #batchnorm
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    #
    model.add(layers.Dense(128,kernel_regularizer=regularizers.l2(reg)))  # kernel_regularizer
    model.add(tf.keras.layers.BatchNormalization(trainable = trainable))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    #softmax output
    model.add(layers.Dense(4,kernel_regularizer=regularizers.l2(reg), activation='softmax'))

    # model.summary()

    return model


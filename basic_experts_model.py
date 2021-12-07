import tensorflow as tf
from tensorflow.keras import layers, models

from ResBlock import ResnetBlock

def ResNet_expert(train_images):
    S = train_images.shape
    model = models.Sequential()
    chl = 64
    # First 3*3*64 stride = 2 cov layer with pooling
    model.add(layers.Conv2D(chl, (3, 3), strides=2, input_shape=S[1:]))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    # ResNet block 64
    model.add(ResnetBlock(3, [0, chl]))

    model.add(ResnetBlock(3, [0, chl]))

    model.add(ResnetBlock(3, [0, chl]))

    model.add(ResnetBlock(3, [0, chl]))

    model.add(ResnetBlock(3, [0, chl]))

    model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    # ResNet block 64
    # model.add(ResnetBlock(3, [0, chl]))
    #
    # model.add(ResnetBlock(3, [0, chl]))
    #
    # model.add(ResnetBlock(3, [0, chl]))
    #
    # model.add(ResnetBlock(3, [0, chl]))
    # 
    # model.add(ResnetBlock(3, [0, chl]))
    #
    # model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(512))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    #
    model.add(layers.Dense(128))  # kernel_regularizer=regularizers.l2(0.001)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(4, activation='softmax'))

    # model.summary()

    return model


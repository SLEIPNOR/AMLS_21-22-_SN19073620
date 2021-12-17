""" Self made module by @ Z.Han"""

""" ResBlock.py module contains 3*3 ResNet Block for ResNet designing, 
parameters are imported as kernel size, filters for instance 
class attribution, call function could import the input tensor 
automatically, you can adjust regularization penalty (reg) """

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import regularizers

class ResnetBlock(Model):

    def __init__(self, kernel_size, filters):
        reg = 0 # regularization
        trainable = True # fix batchnorm parameter or not
        super(ResnetBlock, self).__init__(name='')
        self.filters1, self.filters2 = filters

        # inception conv
        self.conv2a = layers.Conv2D(self.filters1, (1, 1), kernel_regularizer=regularizers.l2(reg))
        self.bn2a = layers.BatchNormalization(trainable = trainable)

        # conv
        self.padding = layers.ZeroPadding2D(padding=(1, 1))
        self.conv2b = layers.Conv2D(self.filters2, kernel_size, kernel_regularizer=regularizers.l2(reg))
        self.bn2b = layers.BatchNormalization(trainable = trainable)
        # conv
        self.conv2c = layers.Conv2D(self.filters2, kernel_size, kernel_regularizer=regularizers.l2(reg))
        self.bn2c = layers.BatchNormalization(trainable = trainable)


    def call(self, tensor):
        # unified channel number
        if self.filters1 != 0:
            tensor = self.conv2a(tensor)

        x = self.padding(tensor)
        x = self.conv2b(x) # conv
        x = self.bn2b(x)
        x = tf.nn.relu(x) # activation

        x = self.padding(x) # padding
        x = self.conv2c(x)


        x += tensor # add

        x = self.bn2c(x) # Batchnorm

        return tf.nn.relu(x)
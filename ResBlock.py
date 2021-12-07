""" Self made module by @ Z.Han"""

""" ResBlock.py module contains 3*3 ResNet Block for ResNet designing, 
parameters are imported as kernel size, filters for instance 
class attribution, call function could import the input tensor 
automatically
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


class ResnetBlock(Model):

    def __init__(self, kernel_size, filters):
        super(ResnetBlock, self).__init__(name='')
        self.filters1, self.filters2 = filters

        self.conv2a = layers.Conv2D(self.filters1, (1, 1))
        self.bn2a = layers.BatchNormalization()

        self.padding = layers.ZeroPadding2D(padding=(1, 1))
        self.conv2b = layers.Conv2D(self.filters2, kernel_size)
        self.bn2b = layers.BatchNormalization()

        self.conv2c = layers.Conv2D(self.filters2, kernel_size)
        self.bn2c = layers.BatchNormalization()


    def call(self, tensor, training=False):
        if self.filters1 != 0:
            tensor = self.conv2a(tensor)

        x = self.padding(tensor)
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.padding(x)
        x = self.conv2c(x)


        x += tensor

        x = self.bn2c(x, training=training)

        return tf.nn.relu(x)
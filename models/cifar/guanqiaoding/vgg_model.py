# model from: https://github.com/GuanqiaoDing/CNN-CIFAR10

from tensorflow.keras.layers import Conv2D, Dense, Activation, \
    BatchNormalization, GlobalAveragePooling2D, MaxPooling2D, Input
from tensorflow.keras import regularizers, initializers
from simulatedFL.models.model import Model

import tensorflow as tf
# total params: 0.27M in vgg-20

momentum = 0.9
epsilon = 1e-5
weight_decay = 1e-4


class VGG(Model):

    def __init__(self, num_classes=10, num_blocks=3, learning_rate=0.1, momentum=0.9, metrics=["accuracy"]):
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.momentum = momentum
        super().__init__(Model.InitializationStates.GLOROT_UNIFORM, learning_rate, metrics)

    def conv_bn_relu(self, x, filters, name, kernel_size=(3, 3), strides=(1, 1)):
        """conv2D + batch normalization + relu activation"""

        x = Conv2D(
            filters, kernel_size,
            strides=strides, padding='same', use_bias=False,
            kernel_initializer=initializers.he_normal(),
            # kernel_regularizer=regularizers.l2(weight_decay),
            name=name + '_conv2D'
        )(x)
        x = BatchNormalization(momentum=momentum, epsilon=epsilon, name=name + '_BN')(x)
        x = Activation('relu', name=name + '_relu')(x)
        return x


    def conv_blocks(self, x, filters, num_blocks, name):
        """two conv, downsampling if dimension not match"""

        for i in range(num_blocks):
            if int(x.shape[-1]) != filters:
                x = self.conv_bn_relu(x, filters, strides=(2, 2), name=name + '_blk{}_conv1'.format(i + 1))
            else:
                x = self.conv_bn_relu(x, filters, name + '_blk{}_conv1'.format(i + 1))
            x = self.conv_bn_relu(x, filters, name + '_blk{}_conv2'.format(i + 1))
        return x


    def get_model(self):
        """sequential model without shortcut, same number of parameters as its resnet counterpart"""

        original_input = Input(shape=(32, 32, 3), name='input')
        # level 0:
        # input: 32x32x3; output: 32x32x16
        x = self.conv_bn_relu(original_input, 16, name='lv0')

        # level 1:
        # input: 32x32x16; output: 32x32x16
        x = self.conv_blocks(x, 16, self.num_blocks, name='lv1')

        # level 2:
        # input: 32x32x16; output: 16x16x32
        x = self.conv_blocks(x, 32, self.num_blocks, name='lv2')

        # level 3:
        # input: 16x16x32; output: 8x8x64
        x = self.conv_blocks(x, 64, self.num_blocks, name='lv3')

        # output
        x = GlobalAveragePooling2D(name='global_pool')(x)
        x = Dense(
            self.num_classes,
            activation='softmax',
            kernel_initializer=initializers.he_normal(),
            # kernel_regularizer=regularizers.l2(weight_decay),
            name='FC'
        )(x)
        model = tf.keras.Model(inputs=original_input, outputs=x)
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=self.metrics)
        return model

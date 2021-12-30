# model from: https://github.com/GuanqiaoDing/CNN-CIFAR10

from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Add, Lambda, \
    Activation, GlobalAveragePooling2D, MaxPooling2D, Concatenate, Input
from tensorflow.keras import regularizers, initializers
from tensorflow.keras import backend as K

import tensorflow as tf

from simulatedFL.models.model import Model

# according to <arXiv:1603.05027>
# use pre-activation
# total params: 0.27M in resnet-20-v2

momentum = 0.9
epsilon = 1e-5
weight_decay = 1e-4

class ResNetV2(Model):

    def __init__(self, num_classes=10, num_blocks=3, learning_rate=0.1, momentum=0.9, metrics=["accuracy"]):
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.momentum = momentum
        super().__init__(Model.InitializationStates.GLOROT_UNIFORM, learning_rate, metrics)

    def conv_layer(self, x, filters, kernel_size, strides, name):
        """conv2D layer"""
        return Conv2D(
            filters, kernel_size,
            strides=strides, padding='same',
            use_bias=False,
            kernel_initializer=initializers.he_normal(),
            # kernel_regularizer=regularizers.l2(weight_decay),
            name=name
        )(x)


    def bn_relu_conv(self, x, filters, kernel_size, strides, name):
        """conv2D block: pre-activation"""

        x = BatchNormalization(momentum=momentum, epsilon=epsilon, name=name + '_BN')(x)
        x = Activation(activation='relu', name=name + '_relu')(x)
        x = self.conv_layer(x, filters, kernel_size, strides, name + '_conv2D')
        return x


    def res_block(self, x, dim, name):
        """residue block: two 3X3 conv2D stacks"""

        input_dim = int(x.shape[-1])

        # shortcut
        identity = x
        if input_dim != dim:    # option A in the original paper
            identity = MaxPooling2D(
                pool_size=(1, 1), strides=(2, 2),
                padding='same',
                name=name + '_shortcut_pool'
            )(identity)

            identity = Lambda(
                lambda y: K.concatenate([y, K.zeros_like(y)]),
                name=name + '_shortcut_zeropad'
            )(identity)

        # residual path
        res = x
        if input_dim != dim:
            res = self.bn_relu_conv(res, dim, (3, 3), (2, 2), name + '_res_conv1')
        else:
            res = self.bn_relu_conv(res, dim, (3, 3), (1, 1), name + '_res_conv1')

        res = self.bn_relu_conv(res, dim, (3, 3), (1, 1), name + '_res_conv2')

        # add identity and residue path
        out = Add(name=name + '_add')([identity, res])
        return out


    def resnet(self):

        original_input = Input(shape=(32, 32, 3), name='input')

        # level 0:
        # input: 32x32x3; output: 32x32x16
        x = self.conv_layer(original_input, 16, (3, 3), (1, 1), 'lv0_conv2D')

        # level 1:
        # input: 32x32x16; output: 32x32x16
        for i in range(self.num_blocks):
            x = self.res_block(x, 16, name='lv1_blk{}'.format(i + 1))

        # level 2:
        # input: 32x32x16; output: 16x16x32
        for i in range(self.num_blocks):
            x = self.res_block(x, 32, name='lv2_blk{}'.format(i + 1))

        # level 3:
        # input: 16x16x32; output: 8x8x64
        for i in range(self.num_blocks):
            x = self.res_block(x, 64, name='lv3_blk{}'.format(i + 1))

        x = BatchNormalization(momentum=momentum, epsilon=epsilon, name='lv3_end_BN')(x)
        x = Activation(activation='relu', name='lv3_end_relu')(x)

        # output
        x = GlobalAveragePooling2D(name='pool')(x)
        x = Dense(
            self.num_classes, activation='softmax',
            kernel_initializer=initializers.he_normal(),
            # kernel_regularizer=regularizers.l2(weight_decay),
            name='FC'
        )(x)

        model = tf.keras.Model(inputs=original_input, outputs=x)
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=self.metrics)
        return model

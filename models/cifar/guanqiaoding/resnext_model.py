# model from: https://github.com/GuanqiaoDing/CNN-CIFAR10

from tensorflow.keras.layers import Conv2D, Dense, Add, Concatenate, \
    Activation, BatchNormalization, GlobalAveragePooling2D, Lambda, MaxPooling2D, Input
from tensorflow.keras import regularizers, initializers
from tensorflow.keras import backend as K
from simulatedFL.models.model import Model

import tensorflow as tf

# according to <arXiv:1611.05431>
#
# Mini-version:
# reduce cardinality to 8 and base_width to 4, also reduce out_dim from 4xdim to 2xdim.
# base residual path width: 4x8, double in each level
#
# Output map size       # layers        dim         out_dim         res total width (cardinality=8)
# 32x32                 3n+1            32          64              4x8
# 16x16                 3n              64          128             8x8
# 8x8                   3n              128         256             16x8
#
# Followed by global average pooling and a dense layer with 10 units.
# Total weighted layers: 9n+2
# total params: 0.37M in resnext-29


cardinality = 8
base_width = 4
momentum = 0.9
epsilon = 1e-5
weight_decay = 5e-4


class ResNext(Model):

    def __init__(self, num_classes=10, num_blocks=3, learning_rate=0.1, momentum=0.9, metrics=["accuracy"]):
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.momentum = momentum
        super().__init__(Model.InitializationStates.GLOROT_UNIFORM, learning_rate, metrics)

    def conv_bn_relu(self, x, filters, kernel_size, strides, has_relu, name):
        """common block: conv2D + batch normalization + (optional) relu activation"""

        x = Conv2D(
            filters, kernel_size,
            strides=strides, padding='same',
            use_bias=False,
            kernel_initializer=initializers.he_normal(),
            # kernel_regularizer=regularizers.l2(weight_decay),
            name=name + '_conv2D'
        )(x)
        x = BatchNormalization(momentum=momentum, epsilon=epsilon, name=name + '_BN')(x)
        if has_relu:
            x = Activation(activation='relu', name=name + '_relu')(x)
        return x


    def res_block(self, x, dim, stride, name):
        """residue block: implement diagram c in the original paper"""

        in_dim = int(x.shape[-1])
        width = dim // 32 * base_width
        out_dim = dim * 2

        # shortcut
        identity = x
        if in_dim != out_dim:
            identity = MaxPooling2D(
                pool_size=(1, 1), strides=(stride, stride),
                padding='same',
                name=name + '_shortcut_pool'
            )(identity)

            identity = Lambda(
                lambda y: K.concatenate([y, K.zeros_like(y)]),
                name=name + '_shortcut_zeropad'
            )(identity)

            # Alternatively, use 1x1 to deal with mismatched dimensions
            # identity = conv_bn_relu(
            #     identity, out_dim,
            #     kernel_size=(1, 1), strides=(1, 1), has_relu=False,
            #     name=name + '_shortcut_1x1'
            # )

        # residual path
        res = self.conv_bn_relu(
            x, width * cardinality,
            kernel_size=(1, 1), strides=(1, 1), has_relu=True,
            name=name + '_res_1x1'
        )
        layers_split = list()
        for i in range(cardinality):
            # split
            partial = Lambda(
                lambda y: y[:, :, :, i * width: (i + 1) * width],
                name=name + '_split_g{}'.format(i + 1)
            )(res)

            # 3x3 conv
            partial = self.conv_bn_relu(
                partial, width,
                kernel_size=(3, 3), strides=(stride, stride), has_relu=True,
                name=name + '_g{}_3x3'.format(i + 1)
            )

            layers_split.append(partial)

        # concatenate and restore dimension
        res = Concatenate(name=name + '_concat')(layers_split)
        res = self.conv_bn_relu(
            res, out_dim,
            kernel_size=(1, 1), strides=(1, 1), has_relu=False,
            name=name + '_1x1'
        )

        # add identity and residue path
        out = Add(name=name + '_add')([identity, res])
        out = Activation(activation='relu', name=name + '_relu')(out)
        return out


    def get_model(self):
        """resnext model: mini-version"""

        original_input = Input(shape=(32, 32, 3), name='input')

        # level 0:
        # input: 32x32x3; output: 32x32x32
        x = self.conv_bn_relu(original_input, 32, (3, 3), (1, 1), has_relu=True, name='lv0')

        # level 1:
        # input: 32x32x32; output: 32x32x64
        for i in range(self.num_blocks):
            x = self.res_block(x, 32, 1, name='lv1_blk{}'.format(i + 1))

        # level 2:
        # input: 32x32x64; output: 16X16X128
        for i in range(self.num_blocks):
            if i == 0:
                x = self.res_block(x, 64, 2, name='lv2_blk{}'.format(i + 1))
            else:
                x = self.res_block(x, 64, 1, name='lv2_blk{}'.format(i + 1))

        # level 3:
        # input: 16x16x128; output: 8X8X256
        for i in range(self.num_blocks):
            if i == 0:
                x = self.res_block(x, 128, 2, name='lv3_blk{}'.format(i + 1))
            else:
                x = self.res_block(x, 128, 1, name='lv3_blk{}'.format(i + 1))

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

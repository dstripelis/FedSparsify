import tensorflow as tf

from simulatedFL.models.model import Model


class BrainAge3DCNN(Model):

    # Def for standard conv block:
    # 1. Conv (3x3) + relu
    # 2. Conv (3x3)
    # 3. Batch normalization
    # 4. Relu
    # 5. Maxpool (3x3)

    # Model def
    def __init__(self, kernel_initializer=Model.InitializationStates.GLOROT_UNIFORM, learning_rate=5e-5,
				 metrics=["mae"], batch_size=1):
        self.original_input = tf.keras.layers.Input(shape=(91, 109, 91, 1), batch_size=batch_size, name='input')
        self.learning_rate = learning_rate
        super(BrainAge3DCNN, self).__init__(kernel_initializer, learning_rate, metrics)

    def get_model(self, training_batch_norm=True, *args, **kwargs):
        def conv_block(inputs, num_filters, scope):
            inputs = tf.keras.layers.Conv3D(num_filters, 3, strides=1, padding="same", name=scope + "_conv")(inputs)
            # since we use BatchNorm as InstanceNorm, we need to keep training=True
            inputs = tf.keras.layers.BatchNormalization(center=False, scale=False, axis=[0, 4])\
                (inputs, training=training_batch_norm)
            # inputs = tf.keras.layers.BatchNormalization(center=False, scale=False, axis=[0, 4])(inputs)
            inputs = tf.keras.layers.MaxPooling3D(2, strides=2, padding="valid", name=scope + "_max_pool")(inputs)
            inputs = tf.nn.relu(inputs, name=scope + "_relu")
            return inputs

        # Series of conv blocks
        inputs = conv_block(self.original_input, 32, "conv_block1")
        inputs = conv_block(inputs, 64, "conv_block2")
        inputs = conv_block(inputs, 128, "conv_block3")
        inputs = conv_block(inputs, 256, "conv_block4")
        inputs = conv_block(inputs, 256, "conv_block5")

        inputs = tf.keras.layers.Conv3D(
            64, 1, strides=1, name="post_conv1")(inputs)
        # since we use BatchNorm as InstanceNorm, we need to keep training=True
        inputs = tf.keras.layers.BatchNormalization(center=False, scale=False, axis=[0, 4])\
            (inputs, training=training_batch_norm)
        # inputs = tf.keras.layers.BatchNormalization(center=False, scale=False, axis=[0, 4])(inputs)
        inputs = tf.nn.relu(inputs, name="post_relu")
        inputs = tf.keras.layers.AveragePooling3D(pool_size=(2, 3, 2))(inputs)

        outputs = tf.keras.layers.Conv3D(
            1, 1, strides=1, name="reg_conv",
            bias_initializer=tf.constant_initializer(62.68))(inputs)
        outputs = tf.squeeze(outputs, axis=[1, 2, 3, 4])

        model = tf.keras.Model(inputs=self.original_input, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
                      loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE),
                      metrics=["mae"])
        return model

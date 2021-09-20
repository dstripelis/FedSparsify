def convolution_block_v2(inputs, num_filters, name):
	with tf.variable_scope(name):
		inputs = tf.keras.layers.Conv3D(num_filters, 3, strides=1, padding="same", name=name + "_conv")(inputs)
		inputs = tf.contrib.layers.instance_norm(inputs, center=False, scale=False)
		inputs = tf.keras.layers.MaxPooling3D(2, strides=2, padding="valid", name=name + "_max_pool")(inputs)
		inputs = tf.nn.relu(inputs, name=name + "_relu")
	return inputs


def infer_ages_v2(images, is_training, distribution_based_training):
	with tf.variable_scope("Brain_Age_Model"):

		inputs = convolution_block_v2(images, 32, "conv_block1")
		inputs = convolution_block_v2(inputs, 64, "conv_block2")
		inputs = convolution_block_v2(inputs, 128, "conv_block3")
		inputs = convolution_block_v2(inputs, 256, "conv_block4")
		inputs = convolution_block_v2(inputs, 256, "conv_block5")

		# Last Layer
		inputs = tf.keras.layers.Conv3D(64, 1, strides=1, name="post_conv1")(inputs)
		inputs = tf.contrib.layers.instance_norm(inputs, center=False, scale=False)
		inputs = tf.nn.relu(inputs, name="post_relu")
		inputs = tf.keras.layers.AveragePooling3D(pool_size=(2, 3, 2), name="post_avg_pool")(inputs)

		# Default rate: 0.5
		drop = tf.layers.dropout(inputs, rate=0.5, training=is_training, name="drop")

		if distribution_based_training:
			# Output prob dist.
			output = tf.layers.dense(drop, 36, activation=tf.nn.softmax, name="predicted_distribution")
			output = tf.squeeze(output)
		else:
			# Regression (MSE).
			output = tf.layers.Conv3D(1, kernel_size=1, strides=1, name="reg_conv",
									  bias_initializer=tf.constant_initializer(62.68))(drop)
			# output = tf.layers.Conv3D(1, kernel_size=1, strides=1, name="reg_conv")(drop)

			output = tf.squeeze(output)

		return output
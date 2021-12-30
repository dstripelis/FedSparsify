import tensorflow as tf

from simulatedFL.models.model import Model

from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.regularizers import l2


class CifarResNet(Model):

	def __init__(self, input_tensor_shape, depth=20, num_stacks=3, num_classes=100,
				 kernel_initializer=Model.InitializationStates.GLOROT_UNIFORM, learning_rate=0.001,
				 momentum=0.9, metrics=["accuracy"]):
		super().__init__(kernel_initializer, learning_rate, metrics)
		self.input_tensor_shape = input_tensor_shape
		self.depth = depth
		self.num_classes = num_classes
		self.num_stacks = num_stacks
		self.momentum = momentum

	@classmethod
	def augment_2d(cls, inputs, rotation=0, horizontal_flip=False, vertical_flip=False):
		"""Apply additive augmentation on 2D data.

		# Arguments
		  rotation: A float, the degree range for rotation (0 <= rotation < 180),
			  e.g. 3 for random image rotation between (-3.0, 3.0).
		  horizontal_flip: A boolean, whether to allow random horizontal flip,
			  e.g. true for 50% possibility to flip image horizontally.
		  vertical_flip: A boolean, whether to allow random vertical flip,
			  e.g. true for 50% possibility to flip image vertically.

		# Returns
		  input data after augmentation, whose shape is the same as its original.
		"""
		if inputs.dtype != tf.float32:
			inputs = tf.image.convert_image_dtype(inputs, dtype=tf.float32)

		with tf.name_scope('augmentation'):
			shp = tf.shape(inputs)
			batch_size, height, width = shp[0], shp[1], shp[2]
			width = tf.cast(width, tf.float32)
			height = tf.cast(height, tf.float32)

			transforms = []
			identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)

			if rotation > 0:
				angle_rad = rotation * 3.141592653589793 / 180.0
				angles = tf.random.uniform([batch_size], -angle_rad, angle_rad)
				f = tf.contrib.image.angles_to_projective_transforms(angles,
																	 height, width)
				transforms.append(f)

			if horizontal_flip:
				coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
				shape = [-1., 0., width, 0., 1., 0., 0., 0.]
				flip_transform = tf.convert_to_tensor(shape, dtype=tf.float32)
				flip = tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1])
				noflip = tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])
				transforms.append(tf.where(coin, flip, noflip))

			if vertical_flip:
				coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
				shape = [1., 0., 0., 0., -1., height, 0., 0.]
				flip_transform = tf.convert_to_tensor(shape, dtype=tf.float32)
				flip = tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1])
				noflip = tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])
				transforms.append(tf.where(coin, flip, noflip))

		if transforms:
			f = tf.contrib.image.compose_transforms(*transforms)
			inputs = tf.contrib.image.transform(inputs, f, interpolation='BILINEAR')
		return inputs

	def resnet_layer(self, inputs, num_filters=16,
					 kernel_size=3,
					 strides=1,
					 activation='relu',
					 batch_normalization=True,
					 conv_first=True):

		conv = Conv2D(num_filters,
					  kernel_size=kernel_size,
					  strides=strides,
					  padding='same',
					  kernel_initializer='he_normal',
					  kernel_regularizer=l2(1e-4),
					  bias_regularizer=l2(1e-4))
		x = inputs
		if conv_first:
			x = conv(x)
			if batch_normalization:
				x = BatchNormalization()(x)
			if activation is not None:
				x = Activation(activation)(x)
		else:
			if batch_normalization:
				x = BatchNormalization()(x)
			if activation is not None:
				x = Activation(activation)(x)
			x = conv(x)
		return x


	def get_model(self):
		if (self.depth - 2) % 6 != 0:
			raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
		num_filters = 16
		num_res_blocks = int((self.depth - 2) / 6)

		inputs = Input(shape=self.input_tensor_shape)
		inputs = self.augment_2d(inputs)
		x = self.resnet_layer(inputs=inputs)
		for stack in range(self.num_stacks):
			for res_block in range(num_res_blocks):
				strides = 1
				if stack > 0 and res_block == 0:
					strides = 2
				y = self.resnet_layer(inputs=x,
									  num_filters=num_filters,
									  strides=strides)
				y = self.resnet_layer(inputs=y,
									  num_filters=num_filters,
									  activation=None)
				if stack > 0 and res_block == 0:
					x = self.resnet_layer(inputs=x,
										  num_filters=num_filters,
										  kernel_size=1,
										  strides=strides,
										  activation=None,
										  batch_normalization=False)
				x = tf.keras.layers.add([x, y])
				x = Activation('relu')(x)
			num_filters *= 2
		x = AveragePooling2D(pool_size=8)(x)
		y = Flatten()(x)
		outputs = Dense(self.num_classes,
						activation='softmax',
						kernel_initializer='he_normal')(y)
		model = tf.keras.Model(inputs=inputs, outputs=outputs)
		model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum),
					  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
					  metrics=self.metrics)
		return model

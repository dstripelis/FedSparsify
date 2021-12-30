import tensorflow as tf

from simulatedFL.models.model import Model


class FashionMnistModel(Model):

	def __init__(self, kernel_initializer=Model.InitializationStates.GLOROT_UNIFORM, learning_rate=0.02,
				 metrics=["accuracy"], kernel_regularizer=None, bias_regularizer=None):
		super().__init__(kernel_initializer, learning_rate, metrics)
		self.kernel_regularizer = kernel_regularizer
		self.bias_regularizer = bias_regularizer

	def get_model(self):
		"""Prepare a simple dense model."""
		Dense = tf.keras.layers.Dense
		Flatten = tf.keras.layers.Flatten

		model = tf.keras.models.Sequential()
		model.add(Flatten(input_shape=(28, 28)))
		model.add(Dense(128, kernel_initializer=self.kernel_initializer, activation="relu",
						kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer))
		model.add(Dense(128, kernel_initializer=self.kernel_initializer, activation="relu",
						kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer))
		model.add(Dense(10, kernel_initializer=self.kernel_initializer, activation="softmax",
						kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer))

		# TODO change the loss to tf.keras.losses.SparseCategoricalCrossentropy - explicit.
		model.compile(
			optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.0),
			loss="sparse_categorical_crossentropy", metrics=self.metrics)

		return model

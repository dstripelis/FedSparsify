import tensorflow as tf

from tensorflow.keras import layers, models
from simulatedFL.models.model import Model


class Cifar10CNN(Model):

	def __init__(self, kernel_initializer=Model.KERAS_INITIALIZER_GLOROT_UNIFORM, learning_rate=0.01,
				 metrics=["accuracy"]):
		super().__init__(kernel_initializer, learning_rate, metrics)


	def get_model(self):
		"""
		Prepare CNN model
		:return:
		"""
		model = tf.keras.applications.ResNet50(
			include_top=True,
			weights=None,
			input_tensor=None,
			input_shape=None,
			pooling=None,
			classes=100,
			kernel_initializer=self.kernel_initializer
		)

		model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.0),
					  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
					  metrics=self.metrics)

		return model
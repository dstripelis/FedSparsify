import tensorflow as tf

from simulatedFL.models.model import Model


class CifarResNet50(Model):

	def __init__(self, kernel_initializer=Model.InitializationStates.GLOROT_UNIFORM, learning_rate=0.01,
				 momentum=0.9, metrics=["accuracy"], classes_num=100):
		super().__init__(kernel_initializer, learning_rate, metrics)
		self.classes_num = classes_num
		self.momentum = momentum


	def get_model(self):
		"""
		Prepare CNN model
		:return:
		"""
		model = tf.keras.applications.ResNet50V2(
			include_top=True,
			weights=None,
			input_tensor=None,
			input_shape=(32, 32, 3),
			pooling=None,
			classes=self.classes_num
		)

		model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum),
					  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
					  metrics=self.metrics)

		return model
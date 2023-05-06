import tensorflow as tf

from tensorflow.keras import layers, models, regularizers
from models.model import Model
from utils.optimizers.fed_prox import FedProx

class CifarCNN(Model):

	def __init__(self, kernel_initializer=Model.InitializationStates.GLOROT_UNIFORM, learning_rate=0.005,
				 metrics=["accuracy"], cifar_10=False, cifar_100=False):
		super().__init__(kernel_initializer, learning_rate, metrics)
		self.cifar_10 = cifar_10
		self.cifar_100 = cifar_100

	def get_model(self):
		"""
		Prepare CNN model
		:return:
		"""
		model = models.Sequential()

		model.add(layers.Conv2D(filters=64, kernel_size=3, input_shape=(32, 32, 3),
								activation='relu', kernel_initializer=self.kernel_initializer, padding='same'))
		model.add(layers.Conv2D(filters=64, kernel_size=3, input_shape=(32, 32, 3),
								activation='relu', kernel_initializer=self.kernel_initializer, padding='same'))
		model.add(layers.MaxPool2D(pool_size=2))

		model.add(layers.BatchNormalization())
		model.add(layers.Conv2D(filters=128, kernel_size=3, activation='relu',
								kernel_initializer=self.kernel_initializer, padding='same'))
		model.add(layers.Conv2D(filters=128, kernel_size=3, activation='relu',
								kernel_initializer=self.kernel_initializer, padding='same'))
		model.add(layers.MaxPool2D(pool_size=2))

		model.add(layers.BatchNormalization())
		model.add(layers.Conv2D(filters=128, kernel_size=3, activation='relu',
								kernel_initializer=self.kernel_initializer, padding='same'))
		model.add(layers.Conv2D(filters=128, kernel_size=3, activation='relu',
								kernel_initializer=self.kernel_initializer, padding='same'))
		model.add(layers.MaxPooling2D((2, 2)))

		model.add(layers.Flatten())
		model.add(layers.Dense(512, activation='relu', kernel_initializer=self.kernel_initializer))
		model.add(layers.Dropout(0.2))

		if self.cifar_10:
			model.add(layers.Dense(10, activation='softmax'))
		elif self.cifar_100:
			model.add(layers.Dense(100, activation='softmax'))

		optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.75)
		# optimizer = FedProx(learning_rate=self.learning_rate, mu=0.001)
		model.compile(optimizer=optimizer,
					  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
					  metrics=self.metrics)
		return model
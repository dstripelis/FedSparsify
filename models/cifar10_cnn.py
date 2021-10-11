import tensorflow as tf

from tensorflow.keras import layers, models, regularizers
from simulatedFL.models.model import Model


class Cifar10CNN(Model):

	def __init__(self, kernel_initializer=Model.KERAS_INITIALIZER_GLOROT_UNIFORM, learning_rate=0.005,
				 metrics=["accuracy"]):
		super().__init__(kernel_initializer, learning_rate, metrics)


	def get_model(self):
		"""
		Prepare CNN model
		:return:
		"""
		# model = models.Sequential()
		# model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same',
		# 						kernel_initializer=self.kernel_initializer))
		# model.add(layers.MaxPooling2D((2, 2)))
		# model.add(layers.Dropout(0.2))
		# model.add(layers.BatchNormalization())
		# model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',
		# 						kernel_initializer=self.kernel_initializer))
		# model.add(layers.MaxPooling2D((2, 2)))
		# model.add(layers.Dropout(0.2))
		# model.add(layers.BatchNormalization())
		# # model.add(layers.Dropout(0.5))
		# model.add(layers.Conv2D(384, (3, 3), activation='relu', padding='same',
		# 						kernel_initializer=self.kernel_initializer))
		# model.add(layers.Conv2D(192, (3, 3), activation='relu', padding='same',
		# 						kernel_initializer=self.kernel_initializer))
		# # model.add(layers.Dropout(0.5))
		# model.add(layers.Flatten())
		# model.add(layers.Dense(192, activation='relu', kernel_initializer=self.kernel_initializer))
		# model.add(layers.Dense(10, kernel_initializer=self.kernel_initializer, activation='softmax'))

		model = models.Sequential()
		model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',
								kernel_initializer=self.kernel_initializer, input_shape=(32, 32, 3)))
		model.add(layers.Conv2D(32, (3, 3), activation='relu',
								kernel_initializer=self.kernel_initializer, padding='same'))
		model.add(layers.MaxPooling2D((2, 2)))
		model.add(layers.Dropout(0.2))
		model.add(layers.BatchNormalization())
		model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=self.kernel_initializer, padding='same'))
		model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=self.kernel_initializer, padding='same'))
		model.add(layers.MaxPooling2D((2, 2)))
		model.add(layers.Dropout(0.2))
		model.add(layers.BatchNormalization())
		model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=self.kernel_initializer, padding='same'))
		model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=self.kernel_initializer, padding='same'))
		model.add(layers.MaxPooling2D((2, 2)))
		model.add(layers.Dropout(0.2))
		model.add(layers.Flatten())
		model.add(layers.Dense(128, activation='relu', kernel_initializer=self.kernel_initializer))
		model.add(layers.Dropout(0.2))
		model.add(layers.Dense(10, activation='softmax'))

		model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.0),
					  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
					  metrics=self.metrics)

		return model
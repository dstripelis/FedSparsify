import tensorflow as tf

from tensorflow.keras import layers, models, regularizers
from fedinit_states.models.model import FedModel


class IMDB_LSTM(FedModel):

	def __init__(self, kernel_initializer=FedModel.KERAS_INITIALIZER_GLOROT_UNIFORM, max_features=25000,
				 learning_rate=0.01, metrics=["accuracy"]):
		self.max_features = max_features  # Only consider the top X words
		super().__init__(kernel_initializer, learning_rate, metrics)

	def get_model(self):
		"""
		Prepare CNN model
		:return:
		"""
		model = models.Sequential()
		# Embed each integer in a 128-dimensional vector
		model.add(layers.Embedding(self.max_features, 128))
		# Add 2 bidirectional LSTMs
		model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True,
												   kernel_initializer=self.kernel_initializer)))
		model.add(layers.Bidirectional(layers.LSTM(64,
												   kernel_initializer=self.kernel_initializer)))
		# Add a classifier
		model.add(layers.Dense(1, activation="sigmoid"))

		model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.0),
					  loss="binary_crossentropy",
					  metrics=self.metrics)

		return model

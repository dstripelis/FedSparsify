from utils.optimizers.fed_prox import FedProx
import tensorflow as tf

from models.model import Model


class FashionMnistModel(Model):

	def __init__(self, kernel_initializer=Model.InitializationStates.GLOROT_UNIFORM, learning_rate=0.02,
				 metrics=["accuracy"], kernel_regularizer=None, bias_regularizer=None,
				 use_sgd=False, use_sgd_with_momentum=False, momentum_factor=0.0,
				 use_fedprox=False, fedprox_mu=0.0):
		super().__init__(kernel_initializer, learning_rate, metrics)
		self.kernel_regularizer = kernel_regularizer
		self.bias_regularizer = bias_regularizer
		self.use_sgd = use_sgd
		self.use_sgd_with_momentum = use_sgd_with_momentum
		self.momentum_factor = momentum_factor
		self.use_fedprox = use_fedprox
		self.fedprox_mu = fedprox_mu

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

		if self.use_sgd:
			model.compile(
				optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.0),
				loss="sparse_categorical_crossentropy", metrics=self.metrics)
		if self.use_sgd_with_momentum:
			if self.momentum_factor == 0.0:
				raise RuntimeError("Need to provide a non-zero value for the momentum attenuation term.")
			# So far we have run experiments with m=0.9.
			model.compile(
				optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum_factor),
				loss="sparse_categorical_crossentropy", metrics=self.metrics)
		if self.use_fedprox:
			if self.fedprox_mu == 0.0:
				raise RuntimeError("Need to provide a non-zero value for the FedProx proximal term.")
			# So far we have run experiments with μ=0.001.
			model.compile(
				optimizer=FedProx(learning_rate=self.learning_rate, mu=self.fedprox_mu),
				loss="sparse_categorical_crossentropy", metrics=self.metrics)

		return model

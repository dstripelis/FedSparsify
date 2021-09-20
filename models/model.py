import abc

class FedModel:

	KERAS_INITIALIZER_RANDOM_NORMAL = "random_normal"
	KERAS_INITIALIZER_RANDOM_UNIFORM = "random_uniform"
	KERAS_INITIALIZER_TRUNCATED_NORMAL = "truncated_normal"
	KERAS_INITIALIZER_GLOROT_NORMAL = "glorot_normal"
	KERAS_INITIALIZER_GLOROT_UNIFORM = "glorot_uniform"
	KERAS_INITIALIZER_HE_NORMAL = "he_normal"
	KERAS_INITIALIZER_HE_UNIFORM = "he_uniform"
	KERAS_INITIALIZER_ZEROS = "zeros"
	KERAS_INITIALIZER_ONES = "ones"

	def __init__(self, kernel_initializer, learning_rate, metrics=list()):
		self.kernel_initializer = kernel_initializer
		self.learning_rate = learning_rate
		self.metrics = metrics

	@abc.abstractmethod
	def get_model(self):
		pass
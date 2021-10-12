import abc

class Model:

	class InitializationStates:
		RANDOM = "random"
		BURNIN_SINGLETON = "burnin_singleton"
		BURNIN_MEAN_CONSENSUS = "burnin_mean_consensus"
		BURNIN_SCALED_CONSENSUS = "burnin_scaled_consensus"
		ROUND_ROBIN = "round_robin"
		TRUNCATED_NORMAL = "truncated_normal"
		GLOROT_NORMAL = "glorot_normal"
		GLOROT_UNIFORM = "glorot_uniform"
		HE_NORMAL = "he_normal"
		HE_UNIFORM = "he_uniform"
		ZEROS = "zeros"
		ONES = "ones"

	def __init__(self, kernel_initializer, learning_rate, metrics=list()):
		self.kernel_initializer = kernel_initializer
		self.learning_rate = learning_rate
		self.metrics = metrics

	@abc.abstractmethod
	def get_model(self):
		pass
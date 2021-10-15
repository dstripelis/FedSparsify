import numpy as np

class PurgeOps:

	def __init__(self):
		pass


class PurgeByThreshold(PurgeOps):

	def __init__(self, compression_fraction):
		super(PurgeByThreshold, self).__init__()
		self.compression_fraction = compression_fraction
		self.threshold = None

	def __call__(self, keras_model, federation_round=None):
		model_weights = keras_model.get_weights()
		flat_params = np.concatenate([matrix.flatten() for matrix in model_weights])
		flat_params.sort()

		if self.compression_fraction > 0.0:
			self.threshold = flat_params[int(self.compression_fraction * len(flat_params))]
			print("Compression Fraction: {}, Threshold: {}".format(self.compression_fraction, self.threshold))

			# If variable value is greater than threshold then preserve, else purge.
			# We convert the boolean mask to 0/1 in order to perform element-wise multiplication with the actual weights.
			# (0: False - discard element), (1: True - preserve element)
			masks = [(np.abs(matrix) > self.threshold).astype(int) for matrix in model_weights]

		else:
			masks = [np.ones(matrix.shape) for matrix in model_weights]

		return masks


class PurgeByThresholdProgressive(PurgeOps):

	def __init__(self, compression_fraction_per_round):
		super(PurgeByThresholdProgressive, self).__init__()
		self.compression_fraction_per_round = compression_fraction_per_round

	def __call__(self, model_vars, federation_round):
		pass

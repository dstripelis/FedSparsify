import numpy as np
from simulatedFL.utils.logger import CustomLogger

class PurgeOps:

	def __init__(self):
		pass

	@classmethod
	def purge_model(cls, model, masks):
		purged_weights = [np.multiply(mx, msk) for mx, msk in zip(model.get_weights(), masks)]
		model.set_weights(purged_weights)


class PurgeByWeightMagnitude(PurgeOps):

	"""
	An implementation of the pruning method suggested in paper:
		Zhu, M., & Gupta, S. (2017). To prune, or not to prune: exploring the efficacy of pruning for model compression.
		https://arxiv.org/pdf/1710.01878.pdf
	"""

	def __init__(self, sparsity_level):
		super(PurgeByWeightMagnitude, self).__init__()
		self.sparsity_level = sparsity_level
		self.threshold = None

	def __call__(self, local_model, global_model=None, federation_round=None):
		model_weights = local_model.get_weights()
		matrices_shapes = [matrix.shape for matrix in model_weights]
		matrices_flattened = [matrix.flatten() for matrix in model_weights]
		flat_params = np.concatenate(matrices_flattened)
		flat_params_abs = np.abs(flat_params)
		flat_params_abs.sort()

		if self.sparsity_level > 0.0:
			self.threshold = flat_params_abs[int(self.sparsity_level * len(flat_params_abs))]
			CustomLogger.info("Sparsity Level: {}, Threshold: {}".format(self.sparsity_level, self.threshold))

			# If variable value is greater than threshold then preserve, else purge.
			# We convert the boolean mask to 0/1 in order to perform element-wise multiplication with the actual weights.
			# (1: True - preserve element), (0: False - discard element)
			#  The following was returning inconsistent number of elements.
			# masks = [(np.abs(matrix) > self.threshold).astype(int) for matrix in model_weights]
			masks = []
			for m_flatten, shape in zip(matrices_flattened, matrices_shapes):
				mask = np.array([1 if np.abs(p) > self.threshold else 0 for p in m_flatten]).reshape(shape)
				masks.append(mask)
		else:
			masks = [np.ones(matrix.shape) for matrix in model_weights]

		return masks


class PurgeByThresholdProgressive(PurgeOps):

	def __init__(self, compression_fraction_per_round):
		super(PurgeByThresholdProgressive, self).__init__()
		self.compression_fraction_per_round = compression_fraction_per_round

	def __call__(self, local_model, global_model=None, federation_round=None):
		pass

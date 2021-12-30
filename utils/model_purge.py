import numpy as np

from simulatedFL.utils.logger import CustomLogger
from simulatedFL.utils.model_state import ModelState


class PurgeOps:

	def __init__(self):
		pass

	@classmethod
	def apply_model_masks(cls, model, masks):
		masked_weights = [np.multiply(mx, msk) for mx, msk in zip(model.get_weights(), masks)]
		model.set_weights(masked_weights)

	def purge_params(self, sparsity_level, matrices_flattened=[], matrices_shapes=[],
					 nnz_threshold=False):
		matrices_sizes = [m.size for m in matrices_flattened]
		flat_params = np.concatenate(matrices_flattened)
		flat_params_abs = np.abs(flat_params)
		if nnz_threshold:
			flat_params_abs = flat_params_abs[flat_params_abs != 0.0]

		masks = []
		flat_params_abs.sort()
		if flat_params_abs.size > 0:
			params_threshold = flat_params_abs[int(sparsity_level * len(flat_params_abs))]
			CustomLogger.info("Sparsity Level: {}, Threshold: {}".format(sparsity_level, params_threshold))
			# If variable value is greater than threshold then preserve, else purge.
			# We convert the boolean mask to 0/1 in order to perform element-wise multiplication with the actual weights.
			# (1: True - preserve element), (0: False - discard element)
			for m_flatten, shape in zip(matrices_flattened, matrices_shapes):
				mask = np.array([1 if np.abs(p) >= params_threshold else 0 for p in m_flatten]).reshape(shape)
				masks.append(mask)
		else:
			# If threshold is not set (e.g., a case where a matrix is all zeroes), then we simply consider all values.
			params_threshold = None
			for m_flatten, shape in zip(matrices_flattened, matrices_shapes):
				mask = np.array([1 for p in m_flatten]).reshape(shape)
				masks.append(mask)

		return params_threshold, masks


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

	def __call__(self, purging_model, global_model=None, federation_round=None):
		model_weights = purging_model.get_weights()
		matrices_shapes = [matrix.shape for matrix in model_weights]
		matrices_flattened = [matrix.flatten() for matrix in model_weights]

		if self.sparsity_level > 0.0:
			self.threshold, masks = self.purge_params(sparsity_level=self.sparsity_level,
													  matrices_flattened=matrices_flattened,
													  matrices_shapes=matrices_shapes,
													  nnz_threshold=False)
		else:
			masks = ModelState.get_model_binary_masks(purging_model)

		return masks


class PurgeByNNZWeightMagnitude(PurgeOps):

	""" This class implements the progressive pruning or multiplicative pruning. At every purging step, the number of
	parameters we purge, is proportional to the total number of NoN-Zero parameters, i.e., sparsity_level * NNZ-params
	whereas in the previous approaches, we purge by considering all parameters - including the zero parameters. """
	def __init__(self, sparsity_level):
		super(PurgeByNNZWeightMagnitude, self).__init__()
		self.sparsity_level = sparsity_level

	def __call__(self, purging_model, global_model=None, federation_round=None):
		model_weights = purging_model.get_weights()
		matrices_shapes = [matrix.shape for matrix in model_weights]
		matrices_flattened = [matrix.flatten() for matrix in model_weights]

		if self.sparsity_level > 0.0:
			self.threshold, masks = self.purge_params(sparsity_level=self.sparsity_level,
													  matrices_flattened=matrices_flattened,
													  matrices_shapes=matrices_shapes,
													  nnz_threshold=True)
		else:
			masks = ModelState.get_model_binary_masks(purging_model)

		return masks


class PurgeByNNZWeightMagnitudeRandom(PurgeOps):

	def __init__(self, sparsity_level, num_params):
		super(PurgeByNNZWeightMagnitudeRandom, self).__init__()
		self.sparsity_level = sparsity_level
		self.num_params = num_params
		self.permutation = np.random.permutation(np.arange(num_params))
		self.non_zero_params = self.num_params
		self.purging_params_num = 0

	def __call__(self, purging_model, global_model=None, federation_round=None):
		model_weights = purging_model.get_weights()
		matrices_shapes = [matrix.shape for matrix in model_weights]
		matrices_flattened = [matrix.flatten() for matrix in model_weights]

		if self.sparsity_level > 0.0:
			purging_elements_num = int(self.sparsity_level * len(self.non_zero_params))
			CustomLogger.info("Sparsity Level: {}, Total Purging Elements: {}"
							  .format(self.sparsity_level, purging_elements_num))
			# Random indices selection.
			self.purging_params_num += purging_elements_num
			zeroing_indices = self.permutation[0: self.purging_params_num]
			self.non_zero_params -= purging_elements_num

			flat_params[zeroing_indices] = 0.0
			matrices_edited = []
			start_idx = 0
			masks = []
			for size in matrices_sizes:
				matrices_edited.append(flat_params[start_idx: start_idx + size])
				start_idx = size
			for m_flatten, shape in zip(matrices_edited, matrices_shapes):
				mask = np.array([1 if p != 0.0 else 0 for p in m_flatten]).reshape(shape)
				masks.append(mask)

		else:
			masks = ModelState.get_model_binary_masks(purging_model)

		return masks


class PurgeByLayerWeightMagnitude(PurgeOps):
	"""
	An implementation of the pruning method suggested in paper:
		Han, S., Pool, J., Tran, J. and Dally, W.J., 2015. Learning both weights and connections for efficient neural networks.
		https://arxiv.org/abs/1506.02626
	"""

	def __init__(self, sparsity_level):
		super(PurgeByLayerWeightMagnitude, self).__init__()
		self.sparsity_level = sparsity_level
		self.threshold = None

	def __call__(self, purging_model, global_model=None, federation_round=None):
		model_weights = purging_model.get_weights()
		matrices_shapes = [matrix.shape for matrix in model_weights]
		matrices_flattened = [matrix.flatten() for matrix in model_weights]

		if self.sparsity_level > 0.0:
			masks = []
			thresholds = []
			for midx, (matrix_flattened, matrix_shape) in enumerate(zip(matrices_flattened, matrices_shapes)):
				CustomLogger.info("MatrixIdx: {}".format(midx))
				m_threshold, m_masks = self.purge_params(sparsity_level=self.sparsity_level,
														 matrices_flattened=[matrix_flattened],
														 matrices_shapes=[matrix_shape],
														 nnz_threshold=False)
				# Function self.purge_params() returns list, thus the indexing.
				m_mask = m_masks[0]
				masks.append(m_mask)
				thresholds.append(m_threshold)
			self.threshold = thresholds
		else:
			masks = ModelState.get_model_binary_masks(purging_model)

		return masks


class PurgeByLayerNNZWeightMagnitude(PurgeOps):
	"""
	An implementation of the pruning method suggested in paper:
		Han, S., Pool, J., Tran, J. and Dally, W.J., 2015. Learning both weights and connections for efficient neural networks.
		https://arxiv.org/abs/1506.02626
	"""

	def __init__(self, sparsity_level):
		super(PurgeByLayerNNZWeightMagnitude, self).__init__()
		self.sparsity_level = sparsity_level
		self.threshold = None

	def __call__(self, purging_model, global_model=None, federation_round=None):
		model_weights = purging_model.get_weights()
		matrices_shapes = [matrix.shape for matrix in model_weights]
		matrices_flattened = [matrix.flatten() for matrix in model_weights]

		if self.sparsity_level > 0.0:
			masks = []
			thresholds = []
			for midx, (matrix_flattened, matrix_shape) in enumerate(zip(matrices_flattened, matrices_shapes)):
				CustomLogger.info("MatrixIdx: {}".format(midx))
				m_threshold, m_masks = self.purge_params(sparsity_level=self.sparsity_level,
														 matrices_flattened=[matrix_flattened],
														 matrices_shapes=[matrix_shape],
														 nnz_threshold=True)
				# Function self.purge_params() returns list, thus the indexing.
				m_mask = m_masks[0]
				masks.append(m_mask)
				thresholds.append(m_threshold)
			self.threshold = thresholds
		else:
			masks = ModelState.get_model_binary_masks(purging_model)

		return masks


class PurgeByWeightMagnitudeGradual(PurgeOps):
	"""
	An implementation of the pruning method suggested in paper:
		Zhu, M., & Gupta, S. (2017). To prune, or not to prune: exploring the efficacy of pruning for model compression.
		https://arxiv.org/pdf/1710.01878.pdf
	"""

	def __init__(self, start_at_iteration, sparsity_level_init, sparsity_level_final,
				 total_iterations, delta_iteration_pruning,
				 centralized_model=False, federated_model=False):
		"""
		Semantically, `iterations` refer to the epoch-level granularity in the 
		centralized case, whereas in the federated case it refers to the federation round. 
		"""
		assert self.centralized_model or self.federated_model
		self.start_at_iteration = start_at_iteration
		self.sparsity_level_init = sparsity_level_init
		self.sparsity_level_final = sparsity_level_final
		self.total_iterations = total_iterations
		self.delta_round_pruning = delta_iteration_pruning
		self.centralized_model = centralized_model
		self.federated_model = federated_model
		super(PurgeByWeightMagnitudeGradual, self).__init__()

	def __call__(self, purging_model, global_model=None, federation_round=None):
		model_weights = purging_model.get_weights()
		matrices_shapes = [matrix.shape for matrix in model_weights]
		matrices_flattened = [matrix.flatten() for matrix in model_weights]

		""" 
		Gradual Pruning Equation:
			s_r = s_f + (s_i - s_f)(1 - (r-r0)/RΔr)^3, with r >= r0
		
		Notation:	
			r: current round
			s_r: sparsity level at current round/iteration/epoch
			s_i: initial sparsity level
			s_f: final sparsity level
			r0: which round to start sparsification/pruning
			R: total number of rounds/iterations/epochs
			Δr: sparsity/prune every Δr rounds/iterations/epochs 
		"""
		sparsity_level_fn = lambda r, si, sf, r0, R, dr: \
			sf + (si - sf) * np.power(1 - np.divide(r - r0, np.multiply(R, dr)), 3)

		sparsity_level = sparsity_level_fn(federation_round, self.sparsity_level_init, self.sparsity_level_final,
										   self.start_at_iteration, self.total_iterations, self.delta_round_pruning)

		if federation_round >= self.start_at_iteration and sparsity_level > 0.0:
			self.threshold, masks = self.purge_params(sparsity_level=sparsity_level,
													  matrices_flattened=matrices_flattened,
													  matrices_shapes=matrices_shapes,
													  nnz_threshold=False)
		else:
			masks = ModelState.get_model_binary_masks(purging_model)

		return masks

	def to_json(self):
		if self.centralized_model:
			return {"start_at_epoch": self.start_at_iteration,
					"sparsity_level_init": self.sparsity_level_init,
					"sparsity_level_final": self.sparsity_level_final,
					"total_epochs": self.total_iterations,
					"delta_epoch_pruning": self.delta_round_pruning}
		if self.federated_model:
			return {"start_at_round": self.start_at_iteration,
					"sparsity_level_init": self.sparsity_level_init,
					"sparsity_level_final": self.sparsity_level_final,
					"total_rounds": self.total_iterations,
					"delta_round_pruning": self.delta_round_pruning}


class PurgeSNIP(PurgeOps):

	def __init__(self, model, sparsity, x, y):
		""" You feed a random input sample (x,y) to the model and
		compute the model masks based on connections saliency. """
		super(PurgeSNIP, self).__init__()
		self.precomputed_masks = self.__snip_purging(model, sparsity, x, y)

	def __call__(self, purging_model, global_model=None, federation_round=None):
		""" Always return the precomputed binary masks returned by SNIP. """
		return self.precomputed_masks

	def __snip_purging(self, model, sparsity, x, y):
		import tensorflow as tf
		x = tf.convert_to_tensor(x)
		y = tf.convert_to_tensor(y)

		with tf.GradientTape() as tape:
			y_pred = model(x)
			loss = model.compiled_loss(y, y_pred)

		trainable_var_names = [v.name for v in model.trainable_variables]
		grads = tape.gradient(loss, model.trainable_weights)
		saliences = [tf.abs(grad * weight) for weight, grad in zip(model.trainable_variables, grads)]
		saliences_flat = tf.concat([tf.reshape(s, -1) for s in saliences], 0)

		k = tf.dtypes.cast(
			tf.math.round(
				tf.dtypes.cast(tf.size(saliences_flat), tf.float32) *
				(1 - sparsity)), tf.int32)
		# print(k,"/",tf.size(saliences_flat))
		values, _ = tf.math.top_k(
			saliences_flat, k=tf.size(saliences_flat)
		)
		current_threshold = tf.gather(values, k - 1)
		# print(current_threshold)
		trainable_var_masks = [np.array(tf.cast(tf.greater_equal(s, current_threshold), dtype=s.dtype))
							   for s in saliences]

		# Now need to combine masks for both trainable and non trainable parameters.
		model_masks = []
		trainable_var_masks_iter = iter(trainable_var_masks)
		for v in model.variables:
			if v.name not in trainable_var_names:
				model_masks.append(np.ones(v.shape))
			else:
				model_masks.append(next(trainable_var_masks_iter))
		return model_masks

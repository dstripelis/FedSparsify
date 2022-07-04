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
					 nnz_threshold=False, descending_order=False):
		matrices_sizes = [m.size for m in matrices_flattened]
		if len(matrices_flattened) > 1:
			flat_params = np.concatenate(matrices_flattened)
		else:
			flat_params = matrices_flattened[0]
		flat_params_abs = np.abs(flat_params)
		if nnz_threshold:
			flat_params_abs = flat_params_abs[flat_params_abs != 0.0]

		masks = []
		if descending_order:
			(-flat_params_abs).sort() # from larger to smaller
		else:
			flat_params_abs.sort() # from smaller to larger
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

	def json(self):
		return None


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

	def __init__(self, sparsity_level, sparsify_every_k_round=1):
		super(PurgeByNNZWeightMagnitude, self).__init__()
		self.sparsity_level = sparsity_level
		self.sparsify_every_k_round = sparsify_every_k_round

	def __call__(self, purging_model, global_model=None, federation_round=None):
		model_weights = purging_model.get_weights()
		matrices_shapes = [matrix.shape for matrix in model_weights]
		matrices_flattened = [matrix.flatten() for matrix in model_weights]

		if self.sparsity_level > 0.0 and federation_round % self.sparsify_every_k_round == 0:
			self.threshold, masks = self.purge_params(sparsity_level=self.sparsity_level,
													  matrices_flattened=matrices_flattened,
													  matrices_shapes=matrices_shapes,
													  nnz_threshold=True)
		else:
			masks = ModelState.get_model_binary_masks(purging_model)

		return masks

	def json(self):
		return {"sparsity_level": self.sparsity_level, "sparsify_every_k_round": self.sparsify_every_k_round}


class PurgeByNNZWeightMagnitudeRandom(PurgeOps):

	def __init__(self, sparsity_level, num_params, sparsify_every_k_round):
		super(PurgeByNNZWeightMagnitudeRandom, self).__init__()
		self.sparsity_level = sparsity_level
		self.non_zero_params = num_params
		self.permutation = np.random.permutation(np.arange(num_params))
		self.purging_elements_num = 0
		self._previous_round_id = -1
		self.sparsify_every_k_round = sparsify_every_k_round

	def __call__(self, purging_model, global_model=None, federation_round=None):
		model_weights = purging_model.get_weights()
		matrices_shapes = [matrix.shape for matrix in model_weights]
		matrices_flattened = [matrix.flatten() for matrix in model_weights]
		flat_params = np.concatenate(matrices_flattened)
		matrices_sizes = [m.size for m in matrices_flattened]

		if self.sparsity_level > 0.0 and federation_round % self.sparsify_every_k_round == 0:

			# Random indices selection.
			if self._previous_round_id != federation_round:
				purging_elements_num = int(self.sparsity_level * self.non_zero_params)
				self.purging_elements_num += purging_elements_num
				self.non_zero_params -= purging_elements_num
				self._previous_round_id = federation_round
			CustomLogger.info("Total Purging Elements: {}".format(self.purging_elements_num))
			zeroing_indices = self.permutation[0: self.purging_elements_num]

			flat_params[zeroing_indices] = 0.0
			matrices_edited = []
			start_idx = 0
			masks = []
			for size in matrices_sizes:
				matrices_edited.append(flat_params[start_idx: start_idx + size])
				start_idx += size
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

	def __init__(self, sparsity_level, sparsify_every_k_round):
		super(PurgeByLayerNNZWeightMagnitude, self).__init__()
		self.sparsity_level = sparsity_level
		self.sparsify_every_k_round = sparsify_every_k_round
		self.threshold = None

	def __call__(self, purging_model, global_model=None, federation_round=None):
		model_weights = purging_model.get_weights()
		matrices_shapes = [matrix.shape for matrix in model_weights]
		matrices_flattened = [matrix.flatten() for matrix in model_weights]

		if self.sparsity_level > 0.0 and federation_round % self.sparsify_every_k_round == 0:
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

	def __init__(self, start_at_round, sparsity_level_init, sparsity_level_final,
				 total_rounds, delta_round_pruning, exponent, purge_per_layer=False,
				 centralized_model=False, federated_model=False):
		"""
		Semantically, `iterations` refer to the epoch-level granularity in the
		centralized case, whereas in the federated case it refers to the federation round.
		"""
		self.start_at_round = start_at_round
		self.sparsity_level_init = sparsity_level_init
		self.sparsity_level_final = sparsity_level_final
		self.total_rounds = total_rounds
		self.delta_round_pruning = delta_round_pruning
		self.exponent = exponent
		self.purge_per_layer = purge_per_layer
		self.centralized_model = centralized_model
		self.federated_model = federated_model
		super(PurgeByWeightMagnitudeGradual, self).__init__()

	def __call__(self, purging_model, global_model=None, federation_round=None):
		""" 
		Gradual Pruning Equation:
			s_t = s_f + (s_i - s_f)(1 - (t-t0)/TΔt)^3, with t >= t0
		
		Notation:	
			s_r: sparsity level at current round/iteration/epoch
			s_i: initial sparsity level
			s_f: final sparsity level
			t: current round
			t0: which round to start sparsification/pruning
			T: total number of rounds/iterations/epochs
			Δt: sparsity/prune every Δt rounds/iterations/epochs
			
			
		Comments:
			increasing the exponent, e.g., 3 to 6, it sparsifies more aggressively at the start of training, 
			which may help to reduce transmission cost. 
		"""
		# --- old gradual pruning function ---
		# sparsity_level_fn = lambda t, si, sf, t0, T, dt, exponent: \
		# 	sf + (si - sf) * np.power(1 - np.divide(t - t0, np.multiply(T, dt)), exponent)
		def sparsity_level_fn(t, si, sf, t0, T, f, exp):
			# if it is not a pruning round then prune based on the previous round that
			# "completely" divides frequency, else if it is a pruning round, then prune
			# using the current round id.
			if t % f != 0:
				t = (t // f) * f
			st = sf + (si - sf) * np.power(1 - np.divide(t - t0, T - t0), exp)
			return st

			# if t % f == 0:
			# 	# if it is a pruning round then prune based on t.
			# 	# e.g., if f == 1, then prune at every round, else prune
			# 	# based on the previous round that "completely" divides frequency
			# 	st = sf + (si - sf) * np.power(1 - np.divide(t - t0, T - t0), exp)
			# else:
			# 	t = (t // f) * f
			# 	st = sf + (si - sf) * np.power(1 - np.divide(t - t0, T - t0), exp)
			# return st

		sparsity_level = sparsity_level_fn(federation_round, self.sparsity_level_init, self.sparsity_level_final,
										   self.start_at_round, self.total_rounds, self.delta_round_pruning,
										   self.exponent)

		if federation_round >= self.start_at_round and sparsity_level > 0.0:
			if self.purge_per_layer:
				model_weights = purging_model.weights
				all_thresholds, all_masks = [], []
				for weight in model_weights:
					if weight.trainable:
						matrix_threshold, matrix_mask = self.purge_params(sparsity_level=sparsity_level,
																		  matrices_flattened=[weight.numpy().flatten()],
																		  matrices_shapes=[weight.shape],
																		  nnz_threshold=False)
						all_thresholds.append(matrix_threshold)
						all_masks.append(matrix_mask[0])
					else:
						all_thresholds.append(None)
						all_masks.append(np.ones(weight.shape))
				self.threshold, masks = all_thresholds, all_masks
			else:
				model_weights = purging_model.weights
				trainable_weights = [w for w in model_weights if w.trainable]
				trainable_weights_flatten = [w.numpy().flatten() for w in trainable_weights]
				trainable_weights_shapes = [w.numpy().shape for w in trainable_weights]
				self.threshold, trainable_weights_masks = self.purge_params(sparsity_level=sparsity_level,
																			matrices_flattened=trainable_weights_flatten,
																			matrices_shapes=trainable_weights_shapes,
																			nnz_threshold=False)
				trainable_weights_masks_iter = iter(trainable_weights_masks)
				non_trainable_weights_masks = [np.ones(w.numpy().shape) for w in model_weights if not w.trainable]
				non_trainable_weights_masks_iter = iter(non_trainable_weights_masks)
				masks = []
				for w in model_weights:
					if w.trainable:
						masks.append(next(trainable_weights_masks_iter))
					else:
						masks.append(next(non_trainable_weights_masks_iter))
		else:
			masks = ModelState.get_model_binary_masks(purging_model)

		return masks

	def json(self):
		if self.centralized_model:
			return {"start_at_epoch": self.start_at_round,
					"sparsity_level_init": self.sparsity_level_init,
					"sparsity_level_final": self.sparsity_level_final,
					"total_epochs": self.total_rounds,
					"delta_epoch_pruning": self.delta_round_pruning}
		if self.federated_model:
			return {"start_at_round": self.start_at_round,
					"sparsity_level_init": self.sparsity_level_init,
					"sparsity_level_final": self.sparsity_level_final,
					"total_rounds": self.total_rounds,
					"delta_round_pruning": self.delta_round_pruning}


class PurgeByWeightMagnitudeRandomGradual(PurgeOps):
	"""
	An implementation of the pruning method suggested in paper:
		Zhu, M., & Gupta, S. (2017). To prune, or not to prune: exploring the efficacy of pruning for model compression.
		https://arxiv.org/pdf/1710.01878.pdf
	"""

	def __init__(self, model, start_at_round, sparsity_level_init, sparsity_level_final,
				 total_rounds, delta_round_pruning, exponent,
				 centralized_model=False, federated_model=False):
		"""
		Semantically, `iterations` refer to the epoch-level granularity in the
		centralized case, whereas in the federated case it refers to the federation round.
		"""
		self.start_at_round = start_at_round
		self.num_params = sum([w.numpy().size for w in model.trainable_weights])
		self.permutation = np.random.permutation(np.arange(self.num_params))
		self.sparsity_level_init = sparsity_level_init
		self.sparsity_level_final = sparsity_level_final
		self.total_rounds = total_rounds
		self.delta_round_pruning = delta_round_pruning
		self.exponent = exponent
		self.centralized_model = centralized_model
		self.federated_model = federated_model
		super(PurgeByWeightMagnitudeRandomGradual, self).__init__()

	def __call__(self, purging_model, global_model=None, federation_round=None):
		""" 
		Gradual Pruning Equation:
			s_t = s_f + (s_i - s_f)(1 - (t-t0)/TΔt)^3, with t >= t0

		Notation:	
			s_r: sparsity level at current round/iteration/epoch
			s_i: initial sparsity level
			s_f: final sparsity level
			t: current round
			t0: which round to start sparsification/pruning
			T: total number of rounds/iterations/epochs
			Δt: sparsity/prune every Δt rounds/iterations/epochs


		Comments:
			increasing the exponent, e.g., 3 to 6, it sparsifies more aggressively at the start of training, 
			which may help to reduce transmission cost. 
		"""

		# --- old gradual pruning function ---
		# sparsity_level_fn = lambda t, si, sf, t0, T, dt, exponent: \
		# 	sf + (si - sf) * np.power(1 - np.divide(t - t0, np.multiply(T, dt)), exponent)
		def sparsity_level_fn(t, si, sf, t0, T, f, exp):
			# if it is not a pruning round then prune based on the previous round that
			# "completely" divides frequency, else if it is a pruning round, then prune
			# using the current round id.
			if t % f != 0:
				t = (t // f) * f
			st = sf + (si - sf) * np.power(1 - np.divide(t - t0, T - t0), exp)
			return st

		# if t % f == 0:
		# 	# if it is a pruning round then prune based on t.
		# 	# e.g., if f == 1, then prune at every round, else prune
		# 	# based on the previous round that "completely" divides frequency
		# 	st = sf + (si - sf) * np.power(1 - np.divide(t - t0, T - t0), exp)
		# else:
		# 	t = (t // f) * f
		# 	st = sf + (si - sf) * np.power(1 - np.divide(t - t0, T - t0), exp)
		# return st

		sparsity_level = sparsity_level_fn(federation_round, self.sparsity_level_init, self.sparsity_level_final,
										   self.start_at_round, self.total_rounds, self.delta_round_pruning,
										   self.exponent)

		if federation_round >= self.start_at_round and sparsity_level > 0.0:

			model_weights = purging_model.weights
			trainable_weights = [w for w in model_weights if w.trainable]
			trainable_weights_flatten = [w.numpy().flatten() for w in trainable_weights]
			trainable_weights_sizes = [m.size for m in trainable_weights_flatten]
			trainable_weights_shapes = [w.numpy().shape for w in trainable_weights]
			purging_elements_num = int(sparsity_level * self.num_params)
			CustomLogger.info("Total Purging Elements: {}".format(purging_elements_num))
			trainable_weights_zeroing_indices = self.permutation[0: purging_elements_num]

			flat_params = np.concatenate(trainable_weights_flatten)
			flat_params[trainable_weights_zeroing_indices] = 0.0
			matrices_edited = []
			start_idx = 0

			non_trainable_weights_masks = [np.ones(w.numpy().shape) for w in model_weights if not w.trainable]
			trainable_weights_masks = []
			for size in trainable_weights_sizes:
				matrices_edited.append(flat_params[start_idx: start_idx + size])
				start_idx += size
			for m_flatten, shape in zip(matrices_edited, trainable_weights_shapes):
				mask = np.array([1 if p != 0.0 else 0 for p in m_flatten]).reshape(shape)
				trainable_weights_masks.append(mask)

			non_trainable_weights_masks_iter = iter(non_trainable_weights_masks)
			trainable_weights_masks_iter = iter(trainable_weights_masks)

			masks = []
			for w in model_weights:
				if w.trainable:
					masks.append(next(trainable_weights_masks_iter))
				else:
					masks.append(next(non_trainable_weights_masks_iter))
		else:
			masks = ModelState.get_model_binary_masks(purging_model)

		return masks

	def json(self):
		if self.centralized_model:
			return {"start_at_epoch": self.start_at_round,
					"sparsity_level_init": self.sparsity_level_init,
					"sparsity_level_final": self.sparsity_level_final,
					"total_epochs": self.total_rounds,
					"delta_epoch_pruning": self.delta_round_pruning}
		if self.federated_model:
			return {"start_at_round": self.start_at_round,
					"sparsity_level_init": self.sparsity_level_init,
					"sparsity_level_final": self.sparsity_level_final,
					"total_rounds": self.total_rounds,
					"delta_round_pruning": self.delta_round_pruning}


class PurgeByWeightMagnitudeStepWise(PurgeOps):

	def __init__(self, start_at_round, sparsity_level_init, sparsity_level_final,
				 total_rounds, delta_round_pruning):
		"""
		Semantically, `iterations` refer to the epoch-level granularity in the
		centralized case, whereas in the federated case it refers to the federation round.
		"""
		self.start_at_round = start_at_round
		self.sparsity_level_init = sparsity_level_init
		self.sparsity_level_final = sparsity_level_final
		self.total_rounds = total_rounds
		self.delta_round_pruning = delta_round_pruning
		super(PurgeByWeightMagnitudeStepWise, self).__init__()

	def __call__(self, purging_model, global_model=None, federation_round=None, binary_mask=None):
		model_weights = purging_model.get_weights()
		matrices_shapes = [matrix.shape for matrix in model_weights]
		matrices_flattened = [matrix.flatten() for matrix in model_weights]

		if binary_mask is not None:
			matrices_flattened = []

		""" 
		Gradual Pruning Equation:
			s_t = s_i +[(s_f - s_i) / ceil(T/dt)] * ceil(t/dt)  with t > 0

		Notation:	
			t: current round
			s_i: initial sparsity level			
			s_f: final sparsity level
			T: total number of rounds/iterations/epochs
			dt: sparsity/prune every Δt rounds/iterations/epochs 
		"""
		sparsity_level_fn = lambda t, s_i, s_f, T, dt: \
			s_i + ((s_f - s_i) / np.ceil(T / dt)) * np.ceil(t / dt)

		sparsity_level = sparsity_level_fn(federation_round, self.sparsity_level_init, self.sparsity_level_final,
										   self.total_rounds, self.delta_round_pruning)

		if federation_round >= self.start_at_round and sparsity_level > 0.0:
			self.threshold, masks = self.purge_params(sparsity_level=sparsity_level,
													  matrices_flattened=matrices_flattened,
													  matrices_shapes=matrices_shapes,
													  nnz_threshold=False)
		else:
			masks = ModelState.get_model_binary_masks(purging_model)

		return masks


	def json(self):
		return {"start_at_round": self.start_at_round,
				"sparsity_level_init": self.sparsity_level_init,
				"sparsity_level_final": self.sparsity_level_final,
				"total_rounds": self.total_rounds,
				"delta_round_pruning": self.delta_round_pruning}


class PurgeByNNZDeviationFromGlobal(PurgeOps):

	"""
	Preserve those weights whose distance from the community is the largest
	"""

	def __init__(self, sparsity_level):
		super(PurgeByNNZDeviationFromGlobal, self).__init__()
		self.sparsity_level = sparsity_level
		self.threshold = None

	def __call__(self, purging_model, global_model=None, federation_round=None):
		matrices_shapes = [matrix.shape for matrix in purging_model.get_weights()]
		pruning_model_matrices_flattened = [matrix.flatten() for matrix in purging_model.get_weights()]
		global_model_matrices_flattened = [matrix.flatten() for matrix in global_model.get_weights()]

		deviation_matrices_flattened = [np.subtract(np.abs(m1), np.abs(m2))
				for m1, m2 in zip(pruning_model_matrices_flattened, global_model_matrices_flattened)]

		if self.sparsity_level > 0.0:
			masks = []
			thresholds = []
			for midx, (matrix_flattened, matrix_shape) in enumerate(zip(deviation_matrices_flattened, matrices_shapes)):
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


class PurgeGrasp(PurgeOps):

	def __init__(self, model, sparsity, x, y):
		""" You feed a random input sample (x,y) to the model and
		compute the model masks based on connections saliency. """
		super(PurgeGrasp, self).__init__()
		self.precomputed_masks = self.__grasp_purging(model, sparsity, x, y)

	def __call__(self, purging_model, global_model=None, federation_round=None):
		""" Always return the precomputed binary masks returned by SNIP. """
		return self.precomputed_masks

	def __grasp_purging(self, model, sparsity, x, y):
		import tensorflow as tf
		x = tf.convert_to_tensor(x)
		y = tf.convert_to_tensor(y)

		# compute gradient
		with tf.GradientTape() as tape:
			y_pred = model(x)
			loss = model.compiled_loss(y, y_pred)

		trainable_var_names = [v.name for v in model.trainable_variables]
		# compute gradient
		grads = tape.gradient(loss, model.trainable_weights)

		# compute hessian-gradient product
		# see https://github.com/tensorflow/tensorflow/blob/f8ca8b7422fb4536821ee89e9d107b26aad7f542/tensorflow/python/eager/benchmarks/resnet50/hvp_test.py#L62
		with tf.GradientTape() as outer_tape:
			with tf.GradientTape() as inner_tape:
				y_pred = model(x)
				loss = model.compiled_loss(y, y_pred)
			inner_grads = inner_tape.gradient(loss, model.trainable_variables)
			# grads = [g.values if isinstance(g, tf.IndexedSlices) else g for g in grads]
			# inner_grads = [g.values if isinstance(g, tf.IndexedSlices) else g for g in inner_grads]
			hessian_grad_product = outer_tape.gradient(
				inner_grads, model.trainable_variables, output_gradients=grads
			)

		# Grasp computes -Hg * \theta (eq 8) and remove the top scoring weights.
		# Here we wil consider the Hg * \theta and remove the lowest scoring weights.
		# The two are equivalient

		saliences = [hg * weight for weight, hg in zip(model.trainable_variables, hessian_grad_product)]
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

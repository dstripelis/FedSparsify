import numpy as np

class MergeOps:

	def __init__(self, scaling_factors):
		self._scaling_factors = scaling_factors

	def __call__(self, *args, **kwargs):
		pass

	def set_scaling_factors(self, new_scaling_factors):
		self._scaling_factors = new_scaling_factors

	def scaling_factors_normalized(self):
		norm_factor = sum(self._scaling_factors)
		norm_learners_scaling_factors = [np.divide(factor, norm_factor) for factor in self._scaling_factors]
		return norm_learners_scaling_factors

	def get_scaling_factors(self):
		return self._scaling_factors

	@classmethod
	def merged_model_with_normalized_masks(cls, models, models_masks):
		model_matrices_shapes = [m.shape for m in models[0].get_weights()]
		model_matrices_num = len(models[0].get_weights())

		# Step-1: Find the normalization factor per parameter/position based on the mask value.
		models_masks_norm_factor = []
		for idx in range(model_matrices_num):
			matrix_masks = [model_masks[idx] for model_masks in models_masks]
			matrix_masks_stacked = np.stack(matrix_masks)
			models_masks_norm_factor.append(np.sum(matrix_masks_stacked, axis=0))

		# Step-2: Using the normalization factor of each position, we divide the matrix value with that factor to get
		# the "probability" contribution of the model's parameter to the community model. Then we multiply, the computed
		# "probability" with the actual parameter value.
		scaled_weights = []
		for model, model_masks in zip(models, models_masks):
			model_masks_normed = [np.nan_to_num(np.divide(m, norm), nan=0.0)
								  for m, norm in zip(model_masks, models_masks_norm_factor)]
			model_weights_flatten = [w.flatten() for w in model.get_weights()]
			model_weights_scaled = [np.multiply(w, m) for w, m in zip(model_weights_flatten, model_masks_normed)]
			scaled_weights.append(model_weights_scaled)

		# Step-3: We add up all rescaled values (parameter_probability * parameter_value) and compute the final model.
		scaled_weights_stacked = np.stack(scaled_weights)
		scaled_weights = np.sum(scaled_weights_stacked, axis=0)

		# Step-4: At the end, we reshape every matrix of the final model to its original shape.
		scaled_weights = [np.reshape(w, shape) for w, shape in zip(scaled_weights, model_matrices_shapes)]

		return scaled_weights


class MergeWeightedAverage(MergeOps):

	def __init__(self, scaling_factors):
		super(MergeWeightedAverage, self).__init__(scaling_factors=scaling_factors)

	def __call__(self, models, models_masks=None, *args, **kwargs):
		num_weights = len(models[0].get_weights())
		scaled_weights = []
		norm_learners_scaling_factors = self.scaling_factors_normalized()
		for j in range(num_weights):
			normalized_weights = [norm_learners_scaling_factors[model_idx] * model.get_weights()[j]
								  for model_idx, model in enumerate(models)]
			normalized_weights_np = np.array(normalized_weights)
			normed_weight = normalized_weights_np.sum(axis=0)
			scaled_weights.append(normed_weight)
		return scaled_weights


class MergeMedian(MergeOps):

	def __init__(self, scaling_factors):
		super(MergeMedian, self).__init__(scaling_factors=scaling_factors)

	def __call__(self, models, models_masks=None, *args, **kwargs):
		num_weights = len(models[0].get_weights())
		median_weights = []
		for j in range(num_weights):
			matrix_shape = models[0].get_weights()[j].shape
			stacked_weights = np.stack([model.get_weights()[j].flatten() for model in models])
			median_weight = np.median(stacked_weights, axis=0)
			median_weights.append(np.reshape(median_weight, matrix_shape))
		return median_weights


class MergeAbs(MergeOps):

	def __init__(self, scaling_factors, by_abs_max=False, by_abs_min=False, discard_zeroes=False):
		if by_abs_max is False and by_abs_min is False:
			raise RuntimeError("One of the two flags need to be true.")
		super(MergeAbs, self).__init__(scaling_factors=scaling_factors)
		self.by_abs_max = by_abs_max
		self.by_abs_min = by_abs_min
		self.discard_zeroes = discard_zeroes

	def __call__(self, models, models_masks=None, *args, **kwargs):
		num_weights = len(models[0].get_weights())
		scaled_weights = []
		for j in range(num_weights):
			matrix_shape = models[0].get_weights()[j].shape
			stacked_weights = np.stack([model.get_weights()[j].flatten() for model in models])
			stacked_weights_signs = np.sign(stacked_weights)
			stacked_weights_abs = np.abs(stacked_weights)
			if self.by_abs_min:
				if self.discard_zeroes:
					# we want the absolute minimum value and therefore we replace 0s with +infinity.
					stacked_weights_abs = np.where(stacked_weights_abs == 0.0, np.inf, stacked_weights_abs)

				# find minimum element - column wise
				weight_min = np.min(stacked_weights_abs, axis=0)

				# find positions of absolute minimum elements - column wise
				stacked_min_idx = np.argmax(stacked_weights_abs == weight_min, axis=0)

				# find the sign of the absolute minimum elements
				stacked_weight_min_sign = stacked_weights_signs[
					stacked_min_idx, np.arange(stacked_weights_signs.shape[1])
				]

				# Set all +infinity values to 0s; extreme case: all values for a particular position were 0s and
				# hence we replaced all 0s with np.inf which resulted to min(np.inf, np.inf, ...) = np.inf
				# since np.inf is not a valid value we need to replace any such values with 0s.
				weight_min = np.where(weight_min == np.inf, 0.0, weight_min)

				# assign the sign of the minimum element
				scaled_weight = np.multiply(weight_min, stacked_weight_min_sign)

			if self.by_abs_max:
				# find maximum element - column wise
				weight_max = np.max(stacked_weights_abs, axis=0)

				# find positions of absolute minimum elements - column wise
				stacked_max_idx = np.argmax(stacked_weights_abs == weight_max, axis=0)

				# find the sign of the absolute minimum elements
				stacked_weight_max_sign = stacked_weights_signs[
					stacked_max_idx, np.arange(stacked_weights_signs.shape[1])
				]

				# assign the sign of the minimum element
				scaled_weight = np.multiply(weight_max, stacked_weight_max_sign)

			scaled_weights.append(np.reshape(scaled_weight, matrix_shape))
		return scaled_weights


class MergeAbsMax(MergeAbs):

	def __init__(self, scaling_factors):
		super(MergeAbsMax, self).__init__(scaling_factors=scaling_factors,
										  by_abs_max=True,
										  by_abs_min=False,
										  discard_zeroes=False)


class MergeAbsMin(MergeAbs):

	def __init__(self, scaling_factors, discard_zeroes=False):
		super(MergeAbsMin, self).__init__(scaling_factors=scaling_factors,
										  by_abs_max=False, by_abs_min=True,
										  discard_zeroes=discard_zeroes)


class MergeWeightedAverageNNZ(MergeOps):

	def __init__(self, scaling_factors):
		super(MergeWeightedAverageNNZ, self).__init__(scaling_factors=scaling_factors)

	def __call__(self, models, models_masks=None, *args, **kwargs):
		# Step-1: We use masks {values: 0, 1}, to see which parameters will contribute to the final model.
		# We then multiply each mask position with the respective scaling factor value.
		models_masks_scaled = []
		for factor, masks in zip(self._scaling_factors, models_masks):
			model_masks_flatten_scaled = [np.multiply(factor, m.flatten()) for m in masks]
			models_masks_scaled.append(model_masks_flatten_scaled)

		final_model = self.__class__.merged_model_with_normalized_masks(models, models_masks_scaled)
		return final_model


class MergeWeightedAverageMajorityVoting(MergeOps):

	def __init__(self, scaling_factors):
		super(MergeWeightedAverageMajorityVoting, self).__init__(scaling_factors=scaling_factors)

	def __call__(self, models, models_masks=None, *args, **kwargs):
		num_models = len(models)
		majority_voting_threshold = np.floor(np.divide(num_models, 2))

		# Step-1: Add up all masks
		masks_sum = models_masks[0]
		for model_masks in models_masks[1:]:
			masks_sum = [np.add(m1.flatten(), m2.flatten()) for m1, m2 in zip(masks_sum, model_masks)]

		masks_majority_voting = []
		for m in masks_sum:
			masks_majority_voting.append([1 if p >= majority_voting_threshold else 0 for p in m])

		# Step-2: We use the "Majority Voting Mask" {values: 0, 1}, to see which parameters will contribute to the final
		# model for every learner and then "scale" each position with the associated scaling factor of each learner.
		# CAUTION: The following considers 0s as part of the weighted average.
		models_masks_scaled = []
		for factor in self._scaling_factors:
			model_masks_flatten_scaled = [np.multiply(factor, m) for m in masks_majority_voting]
			models_masks_scaled.append(model_masks_flatten_scaled)

		# An alternative approach is to discard 0s from the final weighted (MajorityVotingNNZ) average by applying a
		# logical and between the mask of the majority voting and the original model mask.
		# models_masks_scaled = []
		# for idx, factor in enumerate(self._scaling_factors):
		# 	model_masks_flatten_scaled = \
		# 	[np.multiply(factor, mmv*mm) for mmv, mm in zip(masks_majority_voting, models_masks[idx])]
		# 	models_masks_scaled.append(model_masks_flatten_scaled)

		final_model = self.__class__.merged_model_with_normalized_masks(models, models_masks_scaled)

		return final_model


class MergeWeightedAverageNNZMajorityVoting(MergeOps):

	def __init__(self, scaling_factors):
		super(MergeWeightedAverageNNZMajorityVoting, self).__init__(scaling_factors=scaling_factors)

	def __call__(self, models, models_masks=None, *args, **kwargs):
		num_models = len(models)
		majority_voting_threshold = np.floor(np.divide(num_models, 2))

		# Step-1: Add up all masks
		masks_sum = models_masks[0]
		for model_masks in models_masks[1:]:
			masks_sum = [np.add(m1.flatten(), m2.flatten()) for m1, m2 in zip(masks_sum, model_masks)]

		masks_majority_voting = []
		for m in masks_sum:
			masks_majority_voting.append([1 if p >= majority_voting_threshold else 0 for p in m])

		# An alternative approach is to discard 0s from the final weighted (MajorityVotingNNZ) average by applying a
		# logical and between the mask of the majority voting and the original model mask.
		models_masks_scaled = []
		for idx, learner_factor in enumerate(self._scaling_factors):
			flatten_masks = [m.flatten() for m in models_masks[idx]]
			model_masks_flatten_scaled = \
			[np.multiply(learner_factor, mmv*mm) for mmv, mm in zip(masks_majority_voting, flatten_masks)]
			models_masks_scaled.append(model_masks_flatten_scaled)

		final_model = self.__class__.merged_model_with_normalized_masks(models, models_masks_scaled)

		return final_model


class MergeTanh(MergeOps):

	def __init__(self, scaling_factors):
		super(MergeTanh, self).__init__(scaling_factors=scaling_factors)

	def __call__(self, models, models_masks, *args, **kwargs):
		global_model = kwargs['global_model'].get_weights()
		num_weights = len(models[0].get_weights())
		new_weights = []
		for j in range(num_weights):
			matrix_shape = models[0].get_weights()[j].shape
			global_weight = global_model[j].flatten()
			all_weights = [model.get_weights()[j].flatten() for model in models]
			all_weights.append(global_weight)
			stacked_weights = np.stack(all_weights)
			aggregated_update = np.apply_along_axis(self.__local_updates_value_tanh_scaling,
													axis=0, beta=0.25, arr=stacked_weights)
			new_weight = global_weight - aggregated_update
			new_weight_reshaped = new_weight.reshape(matrix_shape)
			new_weights.append(new_weight_reshaped)
		return new_weights

	@classmethod
	def __local_updates_value_tanh_scaling(cls, x, beta=1, epsilon=0.001):
		""" Careful! The assumption here is that the learners local matrices are stacked with the last community matrix.
			Basically the values of the last community matrix are at the end of the stack. """
		x_community = x[-1]
		x_learners = x[:-1]
		x_learners = x_learners[np.logical_not(np.isnan(x_learners))]

		# Pseudogradients!
		x_diff = x_community - x_learners
		x_diff_mean = x_diff.mean()
		x_diff_std = x_diff.std(ddof=-1) + epsilon
		z_abs = np.abs(x_diff - x_diff_mean) / x_diff_std  # (positively) standardized values.
		k = beta * z_abs
		w = (np.exp(k) - np.exp(-k)) / (np.exp(k) + np.exp(-k))
		x_diff_scaled = w * x_diff
		val_scaled = x_diff_scaled.sum() / w.sum()

		if np.isnan(val_scaled):
			val_scaled = x_diff_mean

		return val_scaled


class MergeWeightedPseudoGradients(MergeOps):
	def __init__(self, scaling_factors):
		super(MergeWeightedPseudoGradients, self).__init__(scaling_factors=scaling_factors)

	def __call__(self, models, models_masks=None, *args, **kwargs):
		# num_models = len(models)
		# majority_voting_threshold = np.floor(np.divide(num_models, 2))

		global_weights = kwargs['global_model'].get_weights()
		# A list of lists. The length of the outer list represents the number of
		# models being aggregated and the length of each list model member represents
		# the number of matrices considered in the model.
		models_pseudogradients = kwargs['models_pseudogradients']

		# Scale pseduogradients contribution in the federation
		# based on each learner's scaling factor.
		scaled_models_pseudogradients = []
		norm_learners_scaling_factors = self.scaling_factors_normalized()
		num_pseudogradients_per_model = len(models_pseudogradients[0])
		for j in range(num_pseudogradients_per_model):
			normalized_weights = [norm_learners_scaling_factors[model_idx] * pseudo_model[j]
								  for model_idx, pseudo_model in enumerate(models_pseudogradients)]
			normalized_weights_np = np.array(normalized_weights)
			normed_weight = normalized_weights_np.sum(axis=0)
			scaled_models_pseudogradients.append(normed_weight)

		new_model = [np.add(w, p) for w, p in zip(global_weights, scaled_models_pseudogradients)]
		return new_model






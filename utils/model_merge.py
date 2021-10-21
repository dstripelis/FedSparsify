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


class MergeWeightedAverage(MergeOps):

	def __init__(self, scaling_factors):
		super(MergeWeightedAverage, self).__init__(scaling_factors=scaling_factors)

	def __call__(self, models, model_masks=None):
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

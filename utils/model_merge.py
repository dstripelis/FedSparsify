import numpy as np

class MergeOps:

	@classmethod
	def scaling_factors_normalized(cls, scaling_factors):
		norm_factor = sum(scaling_factors)
		norm_learners_scaling_factors = [np.divide(factor, norm_factor) for factor in scaling_factors]
		return norm_learners_scaling_factors

	@classmethod
	def merge_local_models(cls, lmodels, scaling_factors):
		num_weights = len(lmodels[0].get_weights())
		global_weights = []
		norm_learners_scaling_factors = cls.scaling_factors_normalized(scaling_factors)
		for j in range(num_weights):
			normalized_weights = [norm_learners_scaling_factors[model_idx] * model.get_weights()[j]
								  for model_idx, model in enumerate(lmodels)]
			normalized_weights_np = np.array(normalized_weights)
			normed_weight = normalized_weights_np.sum(axis=0)
			global_weights.append(normed_weight)
		return global_weights
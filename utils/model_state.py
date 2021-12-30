import numpy as np

class ModelState:

	@classmethod
	def count_non_zero_elems(cls, model):
		return sum([np.count_nonzero(matrix) for matrix in model.get_weights()])

	@classmethod
	def get_model_binary_masks(cls, model):
		model_weights = model.get_weights()
		masks = []
		for m in model_weights:
			mask = np.array([0 if float(p) == 0.0 else 1 for p in m.flatten()])
			mask = mask.reshape(m.shape)
			masks.append(mask)
		return masks

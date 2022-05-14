import numpy as np
import tensorflow as tf

class PseudoGradientsCallback(tf.keras.callbacks.Callback):

	def __init__(self, global_model, model_masks=None):
		super(PseudoGradientsCallback, self).__init__()
		self.global_model = global_model
		self.global_masks = model_masks
		self.pseudogradients = list()

	def on_train_begin(self, logs=None):
		pass

	def on_train_end(self, logs=None):
		# Iterate over each matrix, subtract the global matrix values from the
		# local matrix to find the pseudogradients and then apply the global
		# mask to nullify/prune positions that need to remain zero.
		for idx, (m_l, m_g) in enumerate(zip(self.model.get_weights(), self.global_model.get_weights())):
			delta_gradients = np.subtract(m_l, m_g)
			if self.global_masks is not None:
				delta_gradients = np.multiply(delta_gradients, self.global_masks[idx])
			self.pseudogradients.append(delta_gradients)

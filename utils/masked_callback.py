import numpy as np
import tensorflow as tf

class MaskedCallback(tf.keras.callbacks.Callback):

	def __init__(self, model_masks):
		super(MaskedCallback, self).__init__()
		self.model_masks = model_masks

	def apply_model_masks(self):
		model_weights = self.model.get_weights()
		masked_weights = [np.multiply(model_weights[idx], mask) for idx, mask in enumerate(self.model_masks)]
		self.model.set_weights(masked_weights)

	def on_train_begin(self, logs=None):
		pass

	def on_train_end(self, logs=None):
		pass

	def on_epoch_begin(self, epoch, logs=None):
		pass

	def on_epoch_end(self, epoch, logs=None):
		pass

	def on_test_begin(self, logs=None):
		pass

	def on_test_end(self, logs=None):
		pass

	def on_predict_begin(self, logs=None):
		pass

	def on_predict_end(self, logs=None):
		pass

	def on_train_batch_begin(self, batch, logs=None):
		""" Need to apply the mask on every batch. """
		self.apply_model_masks()

	def on_train_batch_end(self, batch, logs=None):
		pass

	def on_test_batch_begin(self, batch, logs=None):
		""" Need to apply the mask on every batch. """
		self.apply_model_masks()

	def on_test_batch_end(self, batch, logs=None):
		pass

	def on_predict_batch_begin(self, batch, logs=None):
		""" Need to apply the mask on every batch. """
		self.apply_model_masks()

	def on_predict_batch_end(self, batch, logs=None):
		pass

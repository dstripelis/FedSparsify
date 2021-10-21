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
		super(MaskedCallback, self).on_train_begin(logs)

	def on_train_end(self, logs=None):
		super(MaskedCallback, self).on_train_end(logs)

	def on_epoch_begin(self, epoch, logs=None):
		super(MaskedCallback, self).on_epoch_begin(epoch, logs)

	def on_epoch_end(self, epoch, logs=None):
		""" Need to apply the mask when training is complete, in order to receive the sparsified model. """
		super(MaskedCallback, self).on_epoch_end(epoch, logs)
		self.apply_model_masks()

	def on_test_begin(self, logs=None):
		""" Need to apply the mask when test starts, in order to evaluate the sparsified model. """
		super(MaskedCallback, self).on_test_begin(logs)
		self.apply_model_masks()

	def on_test_end(self, logs=None):
		super(MaskedCallback, self).on_test_end(logs)

	def on_predict_begin(self, logs=None):
		""" Need to apply the mask when inference starts, in order to use the sparsified model. """
		super(MaskedCallback, self).on_predict_begin(logs)
		self.apply_model_masks()

	def on_predict_end(self, logs=None):
		super(MaskedCallback, self).on_predict_end(logs)

	def on_train_batch_begin(self, batch, logs=None):
		""" Need to apply the mask on every batch. """
		super(MaskedCallback, self).on_train_batch_begin(batch, logs)
		self.apply_model_masks()

	def on_train_batch_end(self, batch, logs=None):
		super(MaskedCallback, self).on_train_batch_end(batch, logs)

	def on_test_batch_begin(self, batch, logs=None):
		""" Need to apply the mask on every batch. """
		super(MaskedCallback, self).on_test_batch_begin(batch, logs)

	def on_test_batch_end(self, batch, logs=None):
		""" Need to apply the mask on every batch. """
		super(MaskedCallback, self).on_test_batch_end(batch, logs)

	def on_predict_batch_begin(self, batch, logs=None):
		""" Need to apply the mask on every batch. """
		super(MaskedCallback, self).on_predict_batch_begin(batch, logs)

	def on_predict_batch_end(self, batch, logs=None):
		""" Need to apply the mask on every batch. """
		super(MaskedCallback, self).on_predict_batch_end(batch, logs)

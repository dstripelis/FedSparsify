import time
import numpy as np

from tensorflow.keras.callbacks import Callback


class ExecutionTimeRecorder(object):

	# Callback class for time history (picked up this solution directly from StackOverflow)
	class TimeHistory(Callback):

		def __init__(self):
			super(Callback, self).__init__()
			self.times = []
			self.epoch_time_start = time.time()

		def on_train_begin(self, logs={}):
			self.times = []

		def on_epoch_begin(self, batch, logs={}):
			self.epoch_time_start = time.time()

		def on_epoch_end(self, batch, logs={}):
			self.times.append(time.time() - self.epoch_time_start)


	# Function to approximate training time with each layer independently trained.
	@classmethod
	def get_average_variable_train_time(cls, model, x, y, batch_size, epochs_num=5):
		"""
		The core idea is to train one by one each layer and run K epochs. In other words,
		freeze K-1 layers, train the remaining layer, and capture its execution time.
		Outer loop is over all model layers.

		:param model:
		:param x:
		:param y:
		:param batch_size:
		:param epochs_num:
		:return:
		"""

		time_callback = cls.TimeHistory()
		# Loop through each layer setting it Trainable and others as non trainable
		results = []
		for i in range(len(model.layers)):

			layer_name = model.layers[i].name  # storing name of layer for printing layer

			# Setting all layers as non-Trainable
			for layer in model.layers:
				layer.trainable = False

			# Setting ith layers as trainable
			model.layers[i].trainable = True

			# Compile
			model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])

			# Fit on a small number of epochs with callback that records time for each epoch
			model.fit(x, y, epochs=epochs_num, batch_size=batch_size, verbose=0, callbacks=[time_callback])

			# Print average of the time for each layer
			print(f"{layer_name}: Approx (avg) train time for {epochs_num} epochs = ", np.average(time_callback.times))

			# Since the execution time of every parameter inside a layer is the same, we iterate
			# over all existing variables/matrices and append the computed execution time.
			for v in model.layers[i].variables:
				results.append(np.average(time_callback.times))

		trainable_vars_times = []
		non_trainable_vars_times = []
		for v, t in zip(model.variables, results):
			if v.trainable:
				trainable_vars_times.append(t)
			else:
				non_trainable_vars_times.append(t)

		return trainable_vars_times, non_trainable_vars_times

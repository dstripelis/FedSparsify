from simulatedFL.models.model import Model
from simulatedFL.utils.data_distribution import PartitioningScheme
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1

import os
import json
import random
import numpy as np
import tensorflow as tf

import simulatedFL.utils.model_merge as merge_ops
import simulatedFL.utils.model_purge as purge_ops

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
np.random.seed(1990)
random.seed(1990)
tf.random.set_seed(1990)


if __name__ == "__main__":

	""" Load the data. """
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
	x_train = (x_train.astype('float32') / 256).reshape(-1, 28, 28, 1)
	x_test = (x_test.astype('float32') / 256).reshape(-1, 28, 28, 1)
	pscheme = PartitioningScheme(x_train=x_train, y_train=y_train, partitions_num=10)
	# x_chunks, y_chunks = pscheme.iid_partition()
	x_chunks, y_chunks = pscheme.non_iid_partition(classes_per_partition=2)

	# np.savez("test.npz", x=x_test, y=y_test)
	for idx, (x_chunk, y_chunk) in enumerate(zip(x_chunks, y_chunks)):
		np.savez("train_{}.npz".format(idx+1), x=x_chunk, y=y_chunk)

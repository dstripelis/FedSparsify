from simulatedFL.utils.data_distribution import PartitioningScheme
from simulatedFL.models.fashion_mnist_fc import FashionMnistModel
from simulatedFL.models.cifar.guanqiaoding.resnet_model import ResNet
from simulatedFL.utils.model_purge import PurgeOps, PurgeSNIP

import os
import random
import numpy as np
import tensorflow as tf

from utils.model_purge import PurgeGrasp

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
np.random.seed(1990)
random.seed(1990)
tf.random.set_seed(1990)


if __name__ == "__main__":

	# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
	# x_train = (x_train.astype('float32') / 256).reshape(-1, 28, 28, 1)
	# x_test = (x_test.astype('float32') / 256).reshape(-1, 28, 28, 1)
	# pscheme = PartitioningScheme(x_train=x_train, y_train=y_train, partitions_num=10)
	# x_chunks, y_chunks = pscheme.iid_partition()
	# x_chunks, y_chunks = pscheme.non_iid_partition(classes_per_partition=2)

	# model = FashionMnistModel(learning_rate=0.02, kernel_regularizer=None, bias_regularizer=None).get_model()

	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
	input_shape = x_train.shape[1:]
	model = ResNet(num_classes=10, num_blocks=3, learning_rate=0.01).get_model()

	channel_mean = np.mean(x_train, axis=(0, 1, 2))
	channel_std = np.std(x_train, axis=(0, 1, 2))
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	for i in range(3):
		x_train[:, :, :, i] = (x_train[:, :, :, i] - channel_mean[i]) / channel_std[i]
		x_test[:, :, :, i] = (x_test[:, :, :, i] - channel_mean[i]) / channel_std[i]
	pscheme = PartitioningScheme(x_train=x_train, y_train=y_train, partitions_num=10)
	x_chunks, y_chunks = pscheme.iid_partition()

	p = np.random.permutation(len(x_chunks[0]))
	x_chunks[0], y_chunks[0] = x_chunks[0][p], y_chunks[0][p]
	masks = PurgeGrasp(model=model, sparsity=0.8, x=x_chunks[0][:100], y=y_chunks[0][:100]).precomputed_masks

	PurgeOps.apply_model_masks(model, masks)
	matrices = [m.flatten() for m in model.get_weights()]
	non_zero_num = len([p for m in matrices for p in m if p != 0.0])
	zero_num = len([p for m in matrices for p in m if p == 0.0])
	print(non_zero_num, zero_num, zero_num/(zero_num+non_zero_num))
	print("done")

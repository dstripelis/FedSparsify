from tensorflow.keras.preprocessing.image import ImageDataGenerator
from simulatedFL.models.cifar.guanqiaoding.resnet_model import ResNet
# from simulatedFL.models.cifar.cifar_cnn import CifarCNN
# from simulatedFL.models.cifar.cifar_resnet import CifarResNet
from simulatedFL.utils.model_training import ModelTraining
from simulatedFL.utils.data_distribution import PartitioningScheme

import simulatedFL.utils.model_merge as merge_ops
import simulatedFL.utils.model_purge as purge_ops

import os
import json
import random
import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
np.random.seed(1990)
random.seed(1990)
tf.random.set_seed(1990)

if __name__ == "__main__":

	num_classes = 10
	if num_classes == 10:
		"""Load the data."""
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
		input_shape = x_train.shape[1:]

		"""Model Definition."""
		# model = CifarResNet(input_tensor_shape=input_shape, depth=50, num_classes=10).get_model
		# model = CifarCNN(cifar_10=True).get_model
		model = ResNet(num_classes=10, num_blocks=10, learning_rate=0.01).get_model

		output_logs_dir = os.path.dirname(__file__) + "/../logs/Cifar10/"
		output_npzarrays_dir = os.path.dirname(__file__) + "/../npzarrays/Cifar10/"
		experiment_template = "Cifar10.rounds_{}.learners_{}.participation_{}.le_{}.compression_{}.sparsificationround_{}.finetuning_{}"
	else:
		"""Load the data."""
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
		input_shape = x_train.shape[1:]

		"""Model Definition."""
		# model = CifarResNet(input_tensor_shape=input_shape, depth=50, num_classes=100).get_model
		# model = CifarCNN(cifar_100=True).get_model
		model = ResNet(num_classes=100, num_blocks=10, learning_rate=0.01).get_model

		output_logs_dir = os.path.dirname(__file__) + "/../logs/Cifar100/"
		output_npzarrays_dir = os.path.dirname(__file__) + "/../npzarrays/Cifar100/"
		experiment_template = "Cifar100.rounds_{}.learners_{}.participation_{}.le_{}.compression_{}.sparsificationround_{}.finetuning_{}"


	model().summary()

	# x_train = x_train.astype('float32') / 255
	# x_test = x_test.astype('float32') / 255

	# normalize data
	channel_mean = np.mean(x_train, axis=(0, 1, 2))
	channel_std = np.std(x_train, axis=(0, 1, 2))
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	for i in range(3):
		x_train[:, :, :, i] = (x_train[:, :, :, i] - channel_mean[i]) / channel_std[i]
		x_test[:, :, :, i] = (x_test[:, :, :, i] - channel_mean[i]) / channel_std[i]

	# y_train = tf.keras.utils.to_categorical(y_train, num_classes)
	# y_test = tf.keras.utils.to_categorical(y_test, num_classes)

	rounds_num = 100
	learners_num_list = [1]
	participation_rates_list = [1]
	# participation_rates_list = [1, 0.5, 0.1]

	# One-Shot Pruning
	sparsity_levels = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
	# start_sparsification_at_round = [1, 5, 10]
	start_sparsification_at_round = [90]

	# Centralized Progressive Pruning
	# sparsity_levels = [0.005, 0.01, 0.02, 0.05]
	# start_sparsification_at_round = [1, 25, 50]

	local_epochs = 1
	fine_tuning_epochs = [0]
	batch_size = 32
	train_with_global_mask = True

	for learners_num in learners_num_list:
		for participation_rate in participation_rates_list:
			for sparsity_level in sparsity_levels:
				for sparsification_round in start_sparsification_at_round:
					for fine_tuning_epoch_num in fine_tuning_epochs:

						# fill in string placeholders
						filled_in_template = experiment_template.format(rounds_num,
																		learners_num,
																		str(participation_rate).replace(".", ""),
																		str(local_epochs),
																		str(sparsity_level).replace(".", ""),
																		str(sparsification_round),
																		fine_tuning_epoch_num)
						output_arrays_dir = output_npzarrays_dir + filled_in_template

						pscheme = PartitioningScheme(x_train=x_train, y_train=y_train, partitions_num=learners_num)
						x_chunks, y_chunks = pscheme.iid_partition()
						# x_chunks, y_chunks = pscheme.non_iid_partition(classes_per_partition=10)
						scaling_factors = [y_chunk.size for y_chunk in y_chunks]

						# Merging Ops.
						merge_op = merge_ops.MergeWeightedAverage(scaling_factors)
						# merge_op = merge_ops.MergeWeightedAverageNNZ(scaling_factors)
						# merge_op = merge_ops.MergeWeightedAverageMajorityVoting(scaling_factors)

						# Purging Ops.
						# purge_op = purge_ops.PurgeByWeightMagnitude(sparsity_level=sparsity_level)
						# purge_op = purge_ops.PurgeByNNZWeightMagnitude(sparsity_level=sparsity_level)
						purge_op = purge_ops.PurgeByLayerWeightMagnitude(sparsity_level=sparsity_level)
						# purge_op = purge_ops.PurgeByLayerNNZWeightMagnitude(sparsity_level=sparsity_level)
						# purge_op = purge_ops.PurgeByWeightMagnitudeGradual(start_at_round=0,
						# 												   sparsity_level_init=0.5,
						# 												   sparsity_level_final=0.85,
						# 												   total_rounds=rounds_num,
						# 												   delta_round_pruning=1)
						# sparsity_level = purge_op.to_json()
						federated_training = ModelTraining.FederatedTraining(merge_op=merge_op,
																			 learners_num=learners_num,
																			 rounds_num=rounds_num,
																			 local_epochs=local_epochs,
																			 learners_scaling_factors=scaling_factors,
																			 participation_rate=participation_rate,
																			 batch_size=batch_size,
																			 purge_op_local=None,
																			 purge_op_global=purge_op,
																			 start_purging_at_round=sparsification_round,
																			 fine_tuning_epochs=fine_tuning_epoch_num,
																			 train_with_global_mask=train_with_global_mask,
																			 start_training_with_global_mask_at_round=sparsification_round,
																			 output_arrays_dir=output_arrays_dir)
						federated_training.execution_stats['federated_environment']['sparsity_level'] = sparsity_level
						federated_training.execution_stats['federated_environment']['data_distribution'] = \
							pscheme.to_json_representation()
						federated_training_results = federated_training.start(get_model_fn=model, x_train_chunks=x_chunks,
																			  y_train_chunks=y_chunks, x_test=x_test,
																			  y_test=y_test, info="Cifar")

						execution_output_filename = output_logs_dir + filled_in_template + ".json"
						with open(execution_output_filename, "w+", encoding='utf-8') as fout:
							json.dump(federated_training_results, fout, ensure_ascii=False, indent=4)

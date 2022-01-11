from tensorflow.keras.preprocessing.image import ImageDataGenerator
from simulatedFL.models.cifar.guanqiaoding.resnet_model import ResNet
from simulatedFL.models.cifar.cifar_cnn import CifarCNN
# from simulatedFL.models.cifar.cifar_resnet import CifarResNet
from simulatedFL.utils.model_training import ModelTraining
from simulatedFL.utils.data_distribution import PartitioningScheme
from simulatedFL.utils.model_state import ModelState

import simulatedFL.utils.model_merge as merge_ops
import simulatedFL.utils.model_purge as purge_ops

import os
import json
import random
import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
np.random.seed(1990)
random.seed(1990)
tf.random.set_seed(1990)

if __name__ == "__main__":

	gpus = tf.config.experimental.list_physical_devices("GPU")
	if gpus:
		try:
			for gpu in gpus:
				# tf.config.experimental.set_memory_growth(gpu, False)
				tf.config.experimental.set_virtual_device_configuration(
					gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12288)],) # 12GBs
		except RuntimeError as e:
			# Visible devices must be set before GPUs have been initialized
			print(e)

	num_classes = 10
	if num_classes == 10:
		"""Load the data."""
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
		input_shape = x_train.shape[1:]

		"""Model Definition."""
		# model = CifarResNet(input_tensor_shape=input_shape, depth=50, num_classes=10).get_model
		model = CifarCNN(cifar_10=True).get_model
		# model = ResNet(num_classes=10, num_blocks=5, learning_rate=0.01).get_model

		output_logs_dir = os.path.dirname(__file__) + "/../logs/Cifar10/"
		output_npzarrays_dir = os.path.dirname(__file__) + "/../npzarrays/Cifar10/"
		# experiment_template = "Cifar10.rounds_{}.learners_{}.participation_{}.le_{}.compression_{}.sparsificationround_{}.finetuning_{}"
	else:
		"""Load the data."""
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
		input_shape = x_train.shape[1:]

		"""Model Definition."""
		# model = CifarResNet(input_tensor_shape=input_shape, depth=50, num_classes=100).get_model
		model = CifarCNN(cifar_100=True).get_model
		# model = ResNet(num_classes=100, num_blocks=5, learning_rate=0.01).get_model

		output_logs_dir = os.path.dirname(__file__) + "/../logs/Cifar100/"
		output_npzarrays_dir = os.path.dirname(__file__) + "/../npzarrays/Cifar100/"
		# experiment_template = "Cifar100.rounds_{}.learners_{}.participation_{}.le_{}.compression_{}.sparsificationround_{}.finetuning_{}"

	# experiment_template = "MV.IID." + experiment_template
	experiment_template = \
		"Cifar10.MV.IID.Layerpruning.rounds_{}.learners_{}.participation_{}.le_{}.compression_{}.sparsificationround_{}.sparsifyevery_{}rounds.finetuning_{}"

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

	rounds_num = 200
	learners_num_list = [10, 100]
	participation_rates_list = [1, 0.1]
	# participation_rates_list = [1, 0.5, 0.1]

	# One-Shot Pruning
	# sparsity_levels = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
	# start_sparsification_at_round = [1, 5, 10]
	# start_sparsification_at_round = [90]

	# Centralized Progressive Pruning
	# sparsity_levels = [0]
	# sparsity_levels = [0.7, 0.8, 0.9]
	start_sparsification_at_round = [1]
	sparsity_levels = [0.02, 0.04]
	sparsification_frequency = [2, 4]

	local_epochs = 4
	fine_tuning_epochs = [0]
	batch_size = 32
	train_with_global_mask = True

	for learners_num, participation_rate in zip(learners_num_list, participation_rates_list):
		for sparsity_level in sparsity_levels:
			for frequency in sparsification_frequency:
				for sparsification_round in start_sparsification_at_round:
					for fine_tuning_epoch_num in fine_tuning_epochs:

						# fill in string placeholders
						filled_in_template = experiment_template.format(rounds_num,
																		learners_num,
																		str(participation_rate).replace(".", ""),
																		str(local_epochs),
																		str(sparsity_level).replace(".", ""),
																		str(sparsification_round),
																		str(frequency),
																		fine_tuning_epoch_num)
						output_arrays_dir = output_npzarrays_dir + filled_in_template

						pscheme = PartitioningScheme(x_train=x_train, y_train=y_train, partitions_num=learners_num)
						x_chunks, y_chunks = pscheme.iid_partition()
						# x_chunks, y_chunks = pscheme.non_iid_partition(classes_per_partition=5)
						scaling_factors = [y_chunk.size for y_chunk in y_chunks]

						# Merging Ops.
						# merge_op = merge_ops.MergeWeightedAverage(scaling_factors)
						# merge_op = merge_ops.MergeWeightedAverageNNZ(scaling_factors)
						merge_op = merge_ops.MergeWeightedAverageMajorityVoting(scaling_factors)

						# Purging Ops.
						# purge_op = purge_ops.PurgeByWeightMagnitude(sparsity_level=sparsity_level)
						# purge_op = purge_ops.PurgeByNNZWeightMagnitude(sparsity_level=sparsity_level,
						# 											   sparsify_every_k_round=frequency)
						# purge_op = purge_ops.PurgeByNNZWeightMagnitudeRandom(sparsity_level=sparsity_level,
						# 													 num_params=model().count_params(),
						# 													 sparsify_every_k_round=frequency)
						# purge_op = purge_ops.PurgeByLayerWeightMagnitude(sparsity_level=sparsity_level)
						purge_op = purge_ops.PurgeByLayerNNZWeightMagnitude(sparsity_level=sparsity_level,
																			sparsify_every_k_round=frequency)
						# purge_op = purge_ops.PurgeByWeightMagnitudeGradual(start_at_round=0,
						# 												   sparsity_level_init=0.5,
						# 												   sparsity_level_final=0.85,
						# 												   total_rounds=rounds_num,
						# 												   delta_round_pruning=1)
						# sparsity_level = purge_op.to_json()
						# randint = random.randint(0, learners_num-1)
						# purge_op = purge_ops.PurgeSNIP(model(),
						# 							   sparsity=sparsity_level,
						# 							   x=x_chunks[randint][:batch_size],
						# 							   y=y_chunks[randint][:batch_size])
						# randint = random.randint(0, learners_num-1)
						# purge_op = purge_ops.PurgeGrasp(model(),
						# 							   sparsity=sparsity_level,
						# 							   x=x_chunks[randint][:batch_size],
						# 							   y=y_chunks[randint][:batch_size])

						federated_training = ModelTraining.FederatedTraining(merge_op=merge_op,
																			 learners_num=learners_num,
																			 rounds_num=rounds_num,
																			 local_epochs=local_epochs,
																			 learners_scaling_factors=scaling_factors,
																			 participation_rate=participation_rate,
																			 batch_size=batch_size,
																			 purge_op_local=purge_op,
																			 purge_op_global=None,
																			 start_purging_at_round=sparsification_round,
																			 fine_tuning_epochs=fine_tuning_epoch_num,
																			 train_with_global_mask=train_with_global_mask,
																			 start_training_with_global_mask_at_round=sparsification_round,
																			 output_arrays_dir=output_arrays_dir)
																			 # precomputed_masks=purge_op.precomputed_masks)
						federated_training.execution_stats['federated_environment']['model_params'] = ModelState.count_non_zero_elems(model())
						federated_training.execution_stats['federated_environment']['sparsity_level'] = sparsity_level
						federated_training.execution_stats['federated_environment']['additional_specs'] = purge_op.json()
						federated_training.execution_stats['federated_environment']['data_distribution'] = pscheme.to_json_representation()
						print(federated_training.execution_stats)
						federated_training_results = federated_training.start(get_model_fn=model, x_train_chunks=x_chunks,
																			  y_train_chunks=y_chunks, x_test=x_test,
																			  y_test=y_test, info="CIFAR")

						execution_output_filename = output_logs_dir + filled_in_template + ".json"
						with open(execution_output_filename, "w+", encoding='utf-8') as fout:
							json.dump(federated_training_results, fout, ensure_ascii=False, indent=4)

from simulatedFL.models.cifar.cifar_cnn import CifarCNN
from simulatedFL.models.cifar.cifar_resnet_v2 import ResNetCifar10
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

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
np.random.seed(1990)
random.seed(1990)
tf.random.set_seed(1990)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


def data_augmentation_fn(*image_label_tuple):
	image, label = image_label_tuple
	image = tf.pad(image, [[4, 4],
						   [4, 4], [0, 0]])
	image = tf.image.random_flip_left_right(image, seed=1990)
	image = tf.image.random_crop(image, [32, 32, 3])
	return image, label

if __name__ == "__main__":

	num_classes = 10
	if num_classes == 10:
		"""Load the data."""
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
		input_shape = x_train.shape[1:]

		"""Model Definition."""
		model = ResNetCifar10(num_layers=56).get_model
		# model = CifarCNN(cifar_10=True).get_model

		output_logs_dir = os.path.dirname(__file__) + "/../logs/Cifar10/test/"
		output_npzarrays_dir = os.path.dirname(__file__) + "/../npzarrays/Cifar10/"

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

	experiment_template = \
		"Cifar10.ResNet.FedSparsifyLocal.NonIID.rounds_{}.learners_{}.participation_{}.le_{}.compression_{}.sparsificationround_{}.sparsifyevery_{}rounds.finetuning_{}"

	model().summary()

	x_test = x_test.astype('float32') / 255
	y_test = y_test.astype('int64')
	x_train = x_train.astype('float32') / 255
	y_train = y_train.astype('int64')

	# normalize data
	test_channel_mean = np.mean(x_train, axis=(0, 1, 2))
	test_channel_std = np.std(x_train, axis=(0, 1, 2))
	train_channel_mean = np.mean(x_train, axis=(0, 1, 2))
	train_channel_std = np.std(x_train, axis=(0, 1, 2))

	for i in range(3):
		x_test[:, :, :, i] = (x_test[:, :, :, i] - test_channel_mean[i]) / test_channel_std[i]
		x_train[:, :, :, i] = (x_train[:, :, :, i] - train_channel_mean[i]) / train_channel_std[i]

	rounds_num = 100
	learners_num_list = [10, 100]
	participation_rates_list = [1, 0.1]
	# participation_rates_list = [1, 0.5, 0.1]

	start_sparsification_at_round = [1]
	sparsity_levels = [0.8, 0.85, 0.9, 0.95, 0.99]
	sparsification_frequency = [1]

	local_epochs = 4
	fine_tuning_epochs = [0]
	batch_size = 128
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
						# x_chunks, y_chunks = pscheme.iid_partition()
						x_chunks, y_chunks = pscheme.non_iid_partition(classes_per_partition=5)
						scaling_factors = [y_chunk.size for y_chunk in y_chunks]

						train_datasets = [tf.data.Dataset.from_tensor_slices((x_t, y_t))
										  for (x_t, y_t) in zip(x_chunks, y_chunks)]
						train_datasets = [
							dataset.map(data_augmentation_fn).shuffle(buffer_size=5000, seed=1990).batch(batch_size)
							for dataset in train_datasets
						]

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
						# purge_op = purge_ops.PurgeByLayerNNZWeightMagnitude(sparsity_level=sparsity_level,
						# 													sparsify_every_k_round=frequency)
						# purge_op = purge_ops.PurgeByWeightMagnitudeRandomGradual(model=model(),
						# 														 start_at_round=sparsification_round,
						# 														 sparsity_level_init=0.0,
						# 														 sparsity_level_final=sparsity_level,
						# 														 total_rounds=rounds_num,
						# 														 delta_round_pruning=frequency,
						# 														 exponent=3,
						# 														 federated_model=True)
						purge_op = purge_ops.PurgeByWeightMagnitudeGradual(start_at_round=sparsification_round,
																		   sparsity_level_init=0.0,
																		   sparsity_level_final=sparsity_level,
																		   total_rounds=rounds_num,
																		   delta_round_pruning=frequency,
																		   exponent=3,
																		   federated_model=True)
						# sparsity_level = purge_op.to_json()
						# randint = random.randint(0, learners_num-1)
						# purge_op = purge_ops.PurgeSNIP(model(),
						# 							   sparsity=sparsity_level,
						# 							   x=x_chunks[randint][:256],
						# 							   y=y_chunks[randint][:256])
						# randint = random.randint(0, learners_num-1)
						# purge_op = purge_ops.PurgeGrasp(model(),
						# 							   sparsity=sparsity_level,
						# 							   x=x_chunks[randint][:batch_size],
						# 							   y=y_chunks[randint][:batch_size])

						x_chunks = train_datasets
						y_chunks = [None] * len(x_chunks)
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

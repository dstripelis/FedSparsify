import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

import numpy as np
# np.random.seed(1990)

import random
# random.seed(1990)

import tensorflow as tf
# tf.random.set_seed(1990)

from models.cifar.cifar_cnn import CifarCNN
from models.cifar.cifar_vgg import CifarVGG
from utils.model_training import ModelTraining
from utils.data_distribution import PartitioningScheme
from utils.model_state import ModelState

import utils.model_merge as merge_ops
import utils.model_purge as purge_ops

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

def data_augmentation_fn(*image_label_tuple):
	image, label = image_label_tuple
	padding = 4
	image_size = 32
	target_size = image_size + padding * 2
	image = tf.image.pad_to_bounding_box(image, padding, padding, target_size, target_size)
	image = tf.image.random_flip_left_right(image, seed=1990)
	image = tf.image.random_crop(image, [32, 32, 3])
	return image, label


if __name__ == "__main__":

	num_classes = 100
	if num_classes == 10:
		"""Load the data."""
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
		input_shape = [None] + list(x_train.shape[1:])

		"""Model Definition."""
		model = CifarCNN(cifar_10=True).get_model
		# model = CifarVGG(input_shape, "VGG-16", True, 1e-2, num_classes=10).get_model

		output_logs_dir = os.path.dirname(__file__) + "/../logs/Cifar10/test/"
		output_npzarrays_dir = os.path.dirname(__file__) + "/../npzarrays/Cifar10/"

	else:
		"""Load the data."""
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
		input_shape = [None] + list(x_train.shape[1:])

		"""Model Definition."""
		# model = CifarCNN(cifar_100=True).get_model
		model = CifarVGG(input_shape, "VGG-16", True, 1e-2, num_classes=100).get_model

		output_logs_dir = os.path.dirname(__file__) + "/../logs/Cifar100/test/"
		output_npzarrays_dir = os.path.dirname(__file__) + "/../npzarrays/Cifar100/"
		# experiment_template = "Cifar100.rounds_{}.learners_{}.participation_{}.le_{}.compression_{}.sparsificationround_{}.finetuning_{}"

	experiment_template = \
		"Cifar100.VGG.OneShotPruning.FineTuning10Rounds.NonIID.rounds_{}.learners_{}.participation_{}.le_{}.compression_{}.sparsificationround_{}.sparsifyevery_{}rounds.finetuning_{}"

	model().summary()

	x_train = x_train.astype('float32') / 255
	y_train = y_train.astype('int64')
	x_test = x_test.astype('float32') / 255
	y_test = y_test.astype('int64')

	# normalize data
	train_channel_mean = np.mean(x_train, axis=(0, 1, 2))
	train_channel_std = np.std(x_train, axis=(0, 1, 2))
	test_channel_mean = np.mean(x_test, axis=(0, 1, 2))
	test_channel_std = np.std(x_test, axis=(0, 1, 2))

	for i in range(3):
		x_train[:, :, :, i] = (x_train[:, :, :, i] - train_channel_mean[i]) / train_channel_std[i]
		x_test[:, :, :, i] = (x_test[:, :, :, i] - test_channel_mean[i]) / test_channel_std[i]

	rounds_num = 100
	learners_num_list = [10, 100]
	participation_rates_list = [1, 0.1]

	sparsity_levels = [0.9, 0.95, 0.99]
	sparsification_frequency = [1]
	start_sparsification_at_round = [90]

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
						x_chunks, y_chunks = pscheme.non_iid_partition(classes_per_partition=int(num_classes/2))
						scaling_factors = [y_chunk.size for y_chunk in y_chunks]

						train_datasets = [
							tf.data.Dataset.from_tensor_slices((x_t, y_t))
							for (x_t, y_t) in zip(x_chunks, y_chunks)
						]
						train_datasets = [
							dataset.map(data_augmentation_fn).shuffle(buffer_size=5000, seed=1990).batch(batch_size)
							for dataset in train_datasets
						]

						# model().fit(train_datasets[0], epochs=100, validation_data=(x_test, y_test), validation_freq=1)
						# Merging Ops.
						merge_op = merge_ops.MergeWeightedAverage(scaling_factors)
						# merge_op = merge_ops.MergeWeightedAverageNNZ(scaling_factors)
						# merge_op = merge_ops.MergeWeightedAverageMajorityVoting(scaling_factors)

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
						# purge_op = purge_ops.PurgeByWeightMagnitudeGradual(start_at_round=sparsification_round,
						# 												   sparsity_level_init=0.0,
						# 												   sparsity_level_final=sparsity_level,
						# 												   total_rounds=rounds_num,
						# 												   delta_round_pruning=frequency,
						# 												   exponent=3,
						# 												   purge_per_layer=False,
						# 												   federated_model=True)

						# OneShot
						purge_op = purge_ops.PurgeByWeightMagnitudeGradual(start_at_round=sparsification_round,
																		   sparsity_level_init=sparsity_level,
																		   sparsity_level_final=sparsity_level,
																		   total_rounds=rounds_num,
																		   delta_round_pruning=frequency,
																		   exponent=3,
																		   purge_per_layer=False,
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
																			 purge_op_local=None,
																			 purge_op_global=purge_op,
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

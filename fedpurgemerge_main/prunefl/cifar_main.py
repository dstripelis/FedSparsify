from simulatedFL.fedpurgemerge_main.prunefl.prunefl_training import PruneFLTraining
from simulatedFL.fedpurgemerge_main.prunefl.var_execution_time import ExecutionTimeRecorder
from simulatedFL.utils.data_distribution import PartitioningScheme
from simulatedFL.models.cifar.cifar_cnn import CifarCNN
from simulatedFL.utils.masked_callback import MaskedCallback
from simulatedFL.models.cifar.cifar_vgg import CifarVGG

import os
import json
import random
import numpy as np
import simulatedFL.utils.model_merge as merge_ops
import simulatedFL.utils.model_purge as purge_ops
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
np.random.seed(1990)
random.seed(1990)
tf.random.set_seed(1990)

def data_augmentation_fn(*image_label_tuple):
	image, label = image_label_tuple
	# image = tf.pad(image, [[4, 4],
	# 					   [4, 4], [0, 0]])
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
		# model = CifarCNN(cifar_10=True).get_model
		model = CifarVGG(input_shape, "VGG-16", True, 1e-2, num_classes=10).get_model

		output_logs_dir = os.path.dirname(__file__) + "/../../logs/Cifar10/"
		output_npzarrays_dir = os.path.dirname(__file__) + "/../../npzarrays/Cifar10/"
		experiment_template = "Cifar10.PruneFL.rounds_{}.learners_{}.participation_{}.le_{}.compression_{}.sparsificationround_{}.finetuning_{}"
	else:
		"""Load the data."""
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
		input_shape = [None] + list(x_train.shape[1:])

		"""Model Definition."""
		# model = CifarCNN(cifar_100=True).get_model
		model = CifarVGG(input_shape, "VGG-16", True, 1e-2, num_classes=100).get_model

		output_logs_dir = os.path.dirname(__file__) + "/../../logs/Cifar100/"
		output_npzarrays_dir = os.path.dirname(__file__) + "/../../npzarrays/Cifar100/"
		experiment_template = "Cifar100.VGG16.PruneFL.NonIID.rounds_{}.learners_{}.participation_{}.le_{}.compression_{}.sparsificationround_{}.finetuning_{}"

	model().summary()

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
	learners_num_list = [10, 100]
	participation_rates_list = [1, 0.1]

	sparsity_levels = [0.0]
	start_sparsification_at_round = [0]

	local_epochs = 4
	fine_tuning_epochs = [0]
	batch_size = 128
	train_with_global_mask = True

	trainable_vars_avg_time, non_trainable_vars_avg_time = \
		ExecutionTimeRecorder.get_average_variable_train_time(model(), model(), x_train, y_train, batch_size, 5)
	print("Trainable Parameters Average Execution Time: ", trainable_vars_avg_time)
	print("Non-Trainable Parameters Average Execution Time: ", non_trainable_vars_avg_time)

	for learners_num, participation_rate in zip(learners_num_list, participation_rates_list):
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

					# Merging Ops.
					merge_op = merge_ops.MergeWeightedAverage(scaling_factors)

					print("Phase 1: Training the model of a random client in order to learn the initial masks.")
					randint = random.randint(0, learners_num - 1)
					random_model = model()
					initial_masks_adjustment_iterations = 5
					initial_masks_adjustment_local_epochs = 10
					precomputed_masks = None
					for i in range(initial_masks_adjustment_iterations):
						if precomputed_masks is None:
							random_model.fit(x_chunks[randint], y_chunks[randint], batch_size, initial_masks_adjustment_local_epochs)
						else:
							purge_ops.PurgeOps.apply_model_masks(random_model, precomputed_masks)
							random_model.fit(x_chunks[randint], y_chunks[randint], batch_size, initial_masks_adjustment_local_epochs,
											 callbacks=[MaskedCallback(model_masks=precomputed_masks)])
						prunefl_training_context = PruneFLTraining \
							.PruneFLReadjustment(trainable_vars_avg_time, non_trainable_vars_avg_time, prune_rate=0.3)
						random_model_gradients = prunefl_training_context \
							.compute_gradients(random_model, x_chunks[randint], y_chunks[randint])
						model_masks = prunefl_training_context.readjust_model_masks(
							random_model, random_model_gradients, federation_round=0, total_rounds=rounds_num)
						precomputed_masks = model_masks

					federated_training = PruneFLTraining.FederatedTraining(merge_op=merge_op,
																		   learners_num=learners_num,
																		   rounds_num=rounds_num,
																		   local_epochs=local_epochs,
																		   learners_scaling_factors=scaling_factors,
																		   participation_rate=participation_rate,
																		   batch_size=batch_size,
																		   purge_op_local=None,
																		   purge_op_global=None,
																		   start_purging_at_round=sparsification_round,
																		   fine_tuning_epochs=fine_tuning_epoch_num,
																		   train_with_global_mask=train_with_global_mask,
																		   start_training_with_global_mask_at_round=sparsification_round,
																		   output_arrays_dir=output_arrays_dir,
																		   precomputed_masks=precomputed_masks,
																		   masks_readjustment_rounds=50,
																		   prunefl_training_context=prunefl_training_context)
					federated_training.execution_stats['federated_environment']['sparsity_level'] = sparsity_level
					federated_training.execution_stats['federated_environment']['data_distribution'] = \
						pscheme.to_json_representation()
					federated_training_results = federated_training.start(get_model_fn=model,
																		  x_train_chunks_as_numpy=x_chunks,
																		  y_train_chunks_as_numpy=y_chunks,
																		  x_train_chunks_as_datasets=train_datasets,
																		  y_train_chunks_as_datasets=[None] * len(x_chunks),
																		  x_test=x_test,
																		  y_test=y_test, info="Cifar")

					execution_output_filename = output_logs_dir + filled_in_template + ".json"
					with open(execution_output_filename, "w+", encoding='utf-8') as fout:
						json.dump(federated_training_results, fout, ensure_ascii=False, indent=4)

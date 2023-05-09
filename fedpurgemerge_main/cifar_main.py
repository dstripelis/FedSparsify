from models.cifar.cifar_cnn import CifarCNN
from models.cifar.cifar_vgg import CifarVGG
from utils.model_training import ModelTraining
from utils.data_distribution import PartitioningScheme
from utils.model_state import ModelState

import utils.model_merge as merge_ops
import utils.model_purge as purge_ops

import argparse
import json
import os
import random

import tensorflow as tf
import numpy as np

np.random.seed(1990)
random.seed(1990)
tf.random.set_seed(1990)

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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

	parser = argparse.ArgumentParser()
	parser.add_argument("--cifar_classes_num", default=10, help="How many CIFAR classes, 10 or 100.")
	parser.add_argument("--iid_distribution", default=True, help="Whether to distribute the data in an IID fashion across clients")
	parser.add_argument("--non_iid_distribution", default=False, help="Whether to distribute the data in an Non-IID fashion across clients")
	parser.add_argument("--non_iid_classes_per_learner", default=2, type=int, help="In the case of the Non-IID data distribution the assigned examples should belong to this total number of classes, e.g., 2, 5 per learner")
	parser.add_argument("--federation_rounds", default=100, type=int, help="For how many rounds to train the federated experiment?")
	parser.add_argument("--learners_num", default=100, type=int, help="How many learners to consider in the federation? 10, 100?")
	parser.add_argument("--participation_rate", default=1, type=float, help="What is the participating rate of clients at every round, e.g., 0.1 (10%)?")
	parser.add_argument("--local_epochs", default=4, type=int, help="For how many epochs to train the global model at each client.")
	parser.add_argument("--fine_tuning_epochs", default=0, type=int, help="Whether to fine tune after local pruning, if so for how many epochs.")
	parser.add_argument("--batch_size", default=128, type=int, help="Training model dataset batch size.")
	parser.add_argument("--train_with_global_mask", default=True, help="Whether to enforce the global model binary mask during local training.")
	parser.add_argument("--start_sparsification_at_round", default=1, type=int, help="When (at which round) to start sparsifying the network?")
	parser.add_argument("--sparsity_level", default=0.9, type=float, help="What is the final degree of sparsification, e.g., 0.9 (90%)?")
	parser.add_argument("--sparsification_frequency", default=1, type=int, help="How often to sparsify, every f rounds.")
	parser.add_argument("--merging_op", default=None, help="One of: FedAvg, FedAvgNNZ, MV")
	parser.add_argument("--purging_op", default=None, help="One of: fedsparsify-global, fedsparsify-local, random, oneshot, snip, grasp")
	args = parser.parse_args()
	print("Provided Model Arguments:\n",args, flush=True)

	if int(args.cifar_classes_num) == 10:
		"""Load the data."""
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
		input_shape = [None] + list(x_train.shape[1:])

		"""Model Definition."""
		model = CifarCNN(cifar_10=True).get_model
		# model = CifarVGG(input_shape, "VGG-16", True, 1e-2, num_classes=10).get_model

		output_logs_dir = os.path.dirname(__file__) + "/../logs/Cifar10/"
		output_npzarrays_dir = os.path.dirname(__file__) + "/../npzarrays/Cifar10/"

	else:
		"""Load the data."""
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
		input_shape = [None] + list(x_train.shape[1:])

		"""Model Definition."""
		# model = CifarCNN(cifar_100=True).get_model
		model = CifarVGG(input_shape, "VGG-16", True, 1e-2, num_classes=100).get_model

		output_logs_dir = os.path.dirname(__file__) + "/../logs/Cifar100/"
		output_npzarrays_dir = os.path.dirname(__file__) + "/../npzarrays/Cifar100/"

	if not os.path.exists(output_logs_dir):
		os.makedirs(output_logs_dir)
	if not os.path.exists(output_npzarrays_dir):
		os.makedirs(output_npzarrays_dir)

	experiment_template = \
		"Cifar{}.Rounds_{}.Learners_{}.Participation_{}.Epochs_{}.FinalSparsity_{}.SparsificationAtRound_{}.SparsifyEvery_{}rounds.Finetuning_{}.json"

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

	# fill in string placeholders
	filled_in_template = experiment_template.format(str(args.cifar_classes_num),
													args.federation_rounds,
													args.learners_num,
													str(args.participation_rate).replace(".", ""),
													str(args.local_epochs),
													str(args.sparsity_level).replace(".", ""),
													str(args.start_sparsification_at_round),
													str(args.sparsification_frequency),
													str(args.fine_tuning_epochs))
	output_arrays_dir = output_npzarrays_dir + filled_in_template

	pscheme = PartitioningScheme(x_train=x_train, y_train=y_train, partitions_num=args.learners_num)

	if args.iid_distribution is True:
		x_chunks, y_chunks = pscheme.iid_partition()
	elif args.non_iid_distribution is True:
		x_chunks, y_chunks = pscheme.non_iid_partition(classes_per_partition=int(args.non_iid_classes_per_learner))
	else:
		raise RuntimeError("Please specify whether you want an IID or Non-IID data distribution.")

	scaling_factors = [y_chunk.size for y_chunk in y_chunks]
	train_datasets = [
		tf.data.Dataset.from_tensor_slices((x_t, y_t))
		for (x_t, y_t) in zip(x_chunks, y_chunks)
	]
	train_datasets = [
		dataset.map(data_augmentation_fn).shuffle(buffer_size=5000, seed=1990).batch(args.batch_size)
		for dataset in train_datasets
	]

	# Merging Ops.
	if args.merging_op == "FedAvg":
		merge_op = merge_ops.MergeWeightedAverage(scaling_factors)
	elif args.merging_op == "FedAvgNNZ":
		merge_op = merge_ops.MergeWeightedAverageNNZ(scaling_factors)
	elif args.merging_op == "MV":
		merge_op = merge_ops.MergeWeightedAverageMajorityVoting(scaling_factors)
	else:
		raise RuntimeError("Need to specify the merging operation.")

	precomputed_masks = None
	if args.purging_op == "fedsparsify-global":
		purge_op = purge_ops.PurgeByWeightMagnitudeGradual(start_at_round=args.sparsification_round,
														   sparsity_level_init=0.0,
														   sparsity_level_final=args.sparsity_level,
														   total_rounds=args.federation_rounds,
														   delta_round_pruning=args.sparsification_frequency,
														   exponent=3,
														   purge_per_layer=False,
														   federated_model=True)
	elif args.purging_op == "fedsparsify-local":
		purge_op = purge_ops.PurgeByWeightMagnitudeGradual(start_at_round=args.sparsification_round,
														   sparsity_level_init=0.0,
														   sparsity_level_final=args.sparsity_level,
														   total_rounds=args.federation_rounds,
														   delta_round_pruning=args.sparsification_frequency,
														   exponent=3,
														   purge_per_layer=False,
														   federated_model=False)
	elif args.purging_op == "random":
		purge_op = purge_ops.PurgeByWeightMagnitudeRandomGradual(model=model(),
																 start_at_round=args.sparsification_round,
																 sparsity_level_init=0.0,
																 sparsity_level_final=args.sparsity_level,
																 total_rounds=args.federation_rounds,
																 delta_round_pruning=args.sparsification_frequency,
																 exponent=3,
																 federated_model=True)
	elif args.purging_op == "oneshot":
		# OneShot
		purge_op = purge_ops.PurgeByWeightMagnitudeGradual(start_at_round=args.sparsification_round,
														   sparsity_level_init=args.sparsity_level,
														   sparsity_level_final=args.sparsity_level,
														   total_rounds=args.federation_rounds,
														   delta_round_pruning=args.sparsification_frequency,
														   exponent=3,
														   purge_per_layer=False,
														   federated_model=True)
	elif args.purging_op == "snip":
		randint = random.randint(0, args.learners_num-1)
		purge_op = purge_ops.PurgeSNIP(model(),
									   sparsity=args.sparsity_level,
									   x=x_chunks[randint][:args.batch_size],
									   y=y_chunks[randint][:args.batch_size])
		precomputed_masks = purge_op.precomputed_masks
	elif args.purging_op == "grasp":
		randint = random.randint(0, args.learners_num-1)
		purge_op = purge_ops.PurgeGrasp(model(),
									   sparsity=args.sparsity_level,
									   x=x_chunks[randint][:args.batch_size],
									   y=y_chunks[randint][:args.batch_size])
		precomputed_masks = purge_op.precomputed_masks
	else:
		raise RuntimeError("Need to specify the purging/pruning operation you want to perform.")

	x_chunks = train_datasets
	y_chunks = [None] * len(x_chunks)
	federated_training = ModelTraining.FederatedTraining(
		merge_op=merge_op,
		learners_num=args.learners_num,
		rounds_num=args.federation_rounds,
		local_epochs=args.local_epochs,
		learners_scaling_factors=scaling_factors,
		participation_rate=args.participation_rate,
		batch_size=args.batch_size,
		purge_op_local=None,
		purge_op_global=purge_op,
		start_purging_at_round=args.start_sparsification_at_round,
		fine_tuning_epochs=args.fine_tuning_epochs,
		train_with_global_mask=args.train_with_global_mask,
		start_training_with_global_mask_at_round=args.start_sparsification_at_round,
		output_arrays_dir=output_arrays_dir,
		precomputed_masks=precomputed_masks)
	federated_training.execution_stats['federated_environment']['model_params'] = \
		ModelState.count_non_zero_elems(model())
	federated_training.execution_stats['federated_environment']['sparsity_level'] = \
		args.sparsity_level
	federated_training.execution_stats['federated_environment']['additional_specs'] = \
		purge_op.json()
	federated_training.execution_stats['federated_environment']['data_distribution'] = \
		pscheme.to_json_representation()
	print(federated_training.execution_stats)
	federated_training_results = federated_training.start(get_model_fn=model, x_train_chunks=x_chunks,
														  y_train_chunks=y_chunks, x_test=x_test,
														  y_test=y_test, info="CIFAR")

	execution_output_filename = output_logs_dir + filled_in_template
	with open(execution_output_filename, "w+", encoding='utf-8') as fout:
		json.dump(federated_training_results, fout, ensure_ascii=False, indent=4)

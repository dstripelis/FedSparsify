from models.model import Model
from models.imdb_lstm import IMDB_LSTM
from utils.model_training import ModelTraining
from utils.data_distribution import PartitioningScheme
from tensorflow.keras import preprocessing
from utils.model_state import ModelState

import utils.model_merge as merge_ops
import utils.model_purge as purge_ops

import os
import json
import random
import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
np.random.seed(1990)
random.seed(1990)
tf.random.set_seed(1990)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":

	# gpus = tf.config.experimental.list_physical_devices("GPU")
	# if gpus:
	# 	try:
	# 		for gpu in gpus:
	# 			# tf.config.experimental.set_memory_growth(gpu, False)
	# 			tf.config.experimental.set_virtual_device_configuration(
	# 				gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=24576)],) # 24GBs
	# 	except RuntimeError as e:
	# 		# Visible devices must be set before GPUs have been initialized
	# 		print(e)

	"""Model Definition."""
	max_features = 25000  # Only consider the top X words
	maxlen = 200  # Only consider the first X words of each movie review
	model = IMDB_LSTM(kernel_initializer=Model.InitializationStates.GLOROT_UNIFORM,
					  max_features=max_features).get_model
	model().summary()

	"""Load the data."""
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=max_features)
	x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
	x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

	output_logs_dir = os.path.dirname(__file__) + "/../logs/IMDB/"
	output_npzarrays_dir = os.path.dirname(__file__) + "/../npzarrays/IMDB/"
	experiment_template = \
		"IMDB.SNIP.05.IID.rounds_{}.learners_{}.participation_{}.le_{}.compression_{}.sparsificationround_{}.sparsifyevery_{}rounds.finetuning_{}"

	rounds_num = 200
	learners_num_list = [10]
	participation_rates_list = [1]

	start_sparsification_at_round = [0]
	sparsity_levels = [0.5]
	sparsification_frequency = [0]

	# One-Shot Pruning
	# sparsity_levels = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
	# sparsity_levels = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
	# start_sparsification_at_round = [1, 5, 10, 90]

	# Centralized Progressive Pruning
	# sparsity_levels = [0.005, 0.01, 0.02, 0.05]
	# start_sparsification_at_round = [1, 25, 50]

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
						# x_chunks, y_chunks = pscheme.non_iid_partition(classes_per_partition=1)
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
						# purge_op = purge_ops.PurgeByLayerNNZWeightMagnitude(sparsity_level=sparsity_level)
						# purge_op = purge_ops.PurgeByWeightMagnitudeGradual(start_at_round=0,
						# 												   sparsity_level_init=0.5,
						# 												   sparsity_level_final=0.85,
						# 												   total_rounds=rounds_num,
						# 												   delta_round_pruning=1)
						# sparsity_level = purge_op.to_json()
						randint = random.randint(0, learners_num-1)
						# For SNIP with sparsities in the range of (0.8 - 0.89) we need to feed the whole dataset
						# in order to find the links/connections in the Bi-LSTM network.
						purge_op = purge_ops.PurgeSNIP(
							model=model(), sparsity=sparsity_level, x=x_chunks[randint], y=y_chunks[randint])
						# purge_op = purge_ops.PurgeGrasp(
						# 	model=model(), sparsity=sparsity_level,
						# 	x=x_chunks[randint][:batch_size], y=y_chunks[randint][:batch_size])

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
																			 output_arrays_dir=output_arrays_dir,
																			 precomputed_masks=purge_op.precomputed_masks)
						federated_training.execution_stats['federated_environment']['model_params'] = ModelState.count_non_zero_elems(model())
						federated_training.execution_stats['federated_environment']['sparsity_level'] = sparsity_level
						federated_training.execution_stats['federated_environment']['additional_specs'] = purge_op.json()
						federated_training.execution_stats['federated_environment']['data_distribution'] = pscheme.to_json_representation()
						print(federated_training.execution_stats)
						federated_training_results = federated_training.start(get_model_fn=model,
																			  x_train_chunks=x_chunks,
																			  y_train_chunks=y_chunks, x_test=x_test,
																			  y_test=y_test, info="IMDB")

						execution_output_filename = output_logs_dir + filled_in_template + ".json"
						with open(execution_output_filename, "w+", encoding='utf-8') as fout:
							json.dump(federated_training_results, fout, ensure_ascii=False, indent=4)

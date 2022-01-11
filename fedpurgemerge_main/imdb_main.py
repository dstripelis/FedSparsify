from simulatedFL.models.model import Model
from simulatedFL.models.imdb_lstm import IMDB_LSTM
from simulatedFL.utils.model_training import ModelTraining
from simulatedFL.utils.data_distribution import PartitioningScheme
from tensorflow.keras import preprocessing

import simulatedFL.utils.model_merge as merge_ops
import simulatedFL.utils.model_purge as purge_ops

import os
import json
import random
import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
np.random.seed(1990)
random.seed(1990)
tf.random.set_seed(1990)

if __name__ == "__main__":

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
	experiment_template = "IMDB.rounds_{}.learners_{}.participation_{}.le_{}.compression_{}.sparsificationround_{}.finetuning_{}"

	rounds_num = 200
	learners_num_list = [10]
	participation_rates_list = [1]

	sparsity_levels = [0]
	start_sparsification_at_round = [1]

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
	train_with_global_mask = False

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
					x_chunks, y_chunks = pscheme.iid_partition()
					# x_chunks, y_chunks = pscheme.non_iid_partition(classes_per_partition=1)
					scaling_factors = [y_chunk.size for y_chunk in y_chunks]

					# Merging Ops.
					merge_op = merge_ops.MergeWeightedAverage(scaling_factors)
					# merge_op = merge_ops.MergeWeightedAverageNNZ(scaling_factors)
					# merge_op = merge_ops.MergeWeightedAverageMajorityVoting(scaling_factors)

					# Purging Ops.
					# purge_op = purge_ops.PurgeByWeightMagnitude(sparsity_level=sparsity_level)
					# purge_op = purge_ops.PurgeByNNZWeightMagnitude(sparsity_level=sparsity_level)
					# purge_op = purge_ops.PurgeByNNZWeightMagnitudeRandom(sparsity_level=sparsity_level)
					# purge_op = purge_ops.PurgeByLayerWeightMagnitude(sparsity_level=sparsity_level)
					# purge_op = purge_ops.PurgeByLayerNNZWeightMagnitude(sparsity_level=sparsity_level)
					# purge_op = purge_ops.PurgeByWeightMagnitudeGradual(start_at_round=0,
					# 												   sparsity_level_init=0.5,
					# 												   sparsity_level_final=0.85,
					# 												   total_rounds=rounds_num,
					# 												   delta_round_pruning=1)
					# sparsity_level = purge_op.to_json()
					randint = random.randint(0, learners_num-1)
					purge_op = purge_ops.PurgeSNIP(model(),
												   sparsity=sparsity_level,
												   x=x_chunks[randint][:batch_size],
												   y=y_chunks[randint][:batch_size])
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
																		 purge_op_local=None,
																		 purge_op_global=None,
																		 start_purging_at_round=sparsification_round,
																		 fine_tuning_epochs=fine_tuning_epoch_num,
																		 train_with_global_mask=train_with_global_mask,
																		 start_training_with_global_mask_at_round=sparsification_round,
																		 output_arrays_dir=output_arrays_dir)
					federated_training.execution_stats['federated_environment']['sparsity_level'] = sparsity_level
					federated_training.execution_stats['federated_environment']['data_distribution'] = \
						pscheme.to_json_representation()
					federated_training_results = federated_training.start(get_model_fn=model,
																		  x_train_chunks=x_chunks,
																		  y_train_chunks=y_chunks, x_test=x_test,
																		  y_test=y_test, info="IMDB")

					execution_output_filename = output_logs_dir + filled_in_template + ".json"
					with open(execution_output_filename, "w+", encoding='utf-8') as fout:
						json.dump(federated_training_results, fout, ensure_ascii=False, indent=4)

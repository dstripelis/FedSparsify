from simulatedFL.models.model import Model
from simulatedFL.fedpurgemerge_main.prunefl.prunefl_training import PruneFLTraining
from simulatedFL.fedpurgemerge_main.prunefl.var_execution_time import ExecutionTimeRecorder
from simulatedFL.models.imdb_lstm import IMDB_LSTM
from simulatedFL.utils.model_training import ModelTraining
from simulatedFL.utils.data_distribution import PartitioningScheme
from simulatedFL.utils.masked_callback import MaskedCallback
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

	output_logs_dir = os.path.dirname(__file__) + "/../../logs/IMDB/"
	output_npzarrays_dir = os.path.dirname(__file__) + "/../../npzarrays/IMDB/"
	experiment_template = "IMDB.PruneFL.rounds_{}.learners_{}.participation_{}.le_{}.compression_{}.sparsificationround_{}.finetuning_{}"

	rounds_num = 200
	learners_num_list = [10, 100]
	participation_rates_list = [1, 0.1]

	sparsity_levels = [0]
	start_sparsification_at_round = [0]

	local_epochs = 4
	fine_tuning_epochs = [0]
	batch_size = 32
	train_with_global_mask = True

	trainable_vars_avg_time, non_trainable_vars_avg_time = \
		ExecutionTimeRecorder.get_average_variable_train_time(model(), x_train, y_train, batch_size, 1)
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
					x_chunks, y_chunks = pscheme.iid_partition()
					# x_chunks, y_chunks = pscheme.non_iid_partition(classes_per_partition=1)
					scaling_factors = [y_chunk.size for y_chunk in y_chunks]

					print("Phase 1: Training the model of a random client in order to learn the initial masks.")
					randint = random.randint(0, learners_num - 1)
					random_model = model()
					initial_masks_adjustment_iterations = 5
					initial_masks_adjustment_local_epochs = 1
					precomputed_masks = None
					for i in range(initial_masks_adjustment_iterations):
						if precomputed_masks is None:
							random_model.fit(x_train, y_train, batch_size, initial_masks_adjustment_local_epochs)
						else:
							purge_ops.PurgeOps.apply_model_masks(random_model, precomputed_masks)
							random_model.fit(x_train, y_train, batch_size, initial_masks_adjustment_local_epochs,
											 callbacks=[MaskedCallback(model_masks=precomputed_masks)])
						prunefl_training_context = PruneFLTraining \
							.PruneFLReadjustment(trainable_vars_avg_time, non_trainable_vars_avg_time, prune_rate=0.3)
						random_model_gradients = prunefl_training_context \
							.compute_gradients(random_model, x_chunks[randint], y_chunks[randint])
						model_masks = prunefl_training_context.readjust_model_masks(
							random_model, random_model_gradients, federation_round=0, total_rounds=rounds_num)
						precomputed_masks = model_masks

					# Merging Ops.
					merge_op = merge_ops.MergeWeightedAverage(scaling_factors)
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
																		  x_train_chunks=x_chunks,
																		  y_train_chunks=y_chunks, x_test=x_test,
																		  y_test=y_test, info="IMDB")

					execution_output_filename = output_logs_dir + filled_in_template + ".json"
					with open(execution_output_filename, "w+", encoding='utf-8') as fout:
						json.dump(federated_training_results, fout, ensure_ascii=False, indent=4)

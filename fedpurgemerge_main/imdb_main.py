from simulatedFL.models.model import Model
from simulatedFL.models.imdb_lstm import IMDB_LSTM
from simulatedFL.utils.model_training import ModelTraining
from simulatedFL.utils.data_distribution import PartitioningScheme
from simulatedFL.utils.model_merge import MergeWeightedAverage
from simulatedFL.utils.model_purge import PurgeByWeightMagnitude
from tensorflow.keras import preprocessing

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
	experiment_template = "IMDB.rounds_{}.learners_{}.participation_{}.le_{}.compression_{}.finetuning_{}"

	rounds_num = 300
	learners_num_list = [10]
	participation_rates_list = [1]
	# participation_rates_list = [1, 0.5, 0.1]
	sparsity_levels = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
	local_epochs = 4
	fine_tuning_epochs = [0]

	for learners_num in learners_num_list:
		for participation_rate in participation_rates_list:
			for sparsity_level in sparsity_levels:
				for fine_tuning_epoch_num in fine_tuning_epochs:

					# Fill in placeholders
					filled_in_template = experiment_template.format(rounds_num,
																	learners_num,
																	str(participation_rate).replace(".", ""),
																	str(local_epochs),
																	str(sparsity_level).replace(".", ""),
																	fine_tuning_epoch_num)
					output_arrays_dir = output_npzarrays_dir + filled_in_template

					pscheme = PartitioningScheme(x_train=x_train, y_train=y_train, partitions_num=learners_num)
					x_chunks, y_chunks = pscheme.iid_partition()
					# x_chunks, y_chunks = pscheme.non_iid_partition(classes_per_partition=10)

					scaling_factors = [y_chunk.size for y_chunk in y_chunks]
					merge_op = MergeWeightedAverage(scaling_factors)
					purge_op = PurgeByWeightMagnitude(sparsity_level=sparsity_level)
					federated_training = ModelTraining.FederatedTraining(merge_op=merge_op,
																		 learners_num=learners_num,
																		 rounds_num=rounds_num,
																		 local_epochs=local_epochs,
																		 learners_scaling_factors=scaling_factors,
																		 participation_rate=participation_rate,
																		 batch_size=32,
																		 purge_op_local=purge_op,
																		 fine_tuning_epochs=fine_tuning_epoch_num,
																		 output_arrays_dir=output_arrays_dir)
					federated_training.execution_stats['federated_environment']['sparsity_level'] = sparsity_level
					federated_training.execution_stats['federated_environment']['data_distribution'] = \
						pscheme.to_json_representation()
					federated_training_results = federated_training.start(get_model_fn=model, x_train_chunks=x_chunks,
																		  y_train_chunks=y_chunks, x_test=x_test,
																		  y_test=y_test, info="IMDB")

					execution_output_filename = output_logs_dir + filled_in_template + ".json"
					with open(execution_output_filename, "w+", encoding='utf-8') as fout:
						json.dump(federated_training_results, fout, ensure_ascii=False, indent=4)

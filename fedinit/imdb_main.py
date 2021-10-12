from simulatedFL.models.model import Model
from simulatedFL.models.imdb_lstm import IMDB_LSTM
from simulatedFL.utils.model_training import ModelTraining
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
	model = IMDB_LSTM(kernel_initializer=Model.InitializationStates.GLOROT_UNIFORM, max_features=max_features).get_model
	model().summary()

	"""Load the data."""
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=max_features)
	x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
	x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

	output_filename_template = "fedinit_states/logs/IMDB/IMDB.rounds_{}.learners_{}.participation_{}.init_{}.burnin_{}.json"
	rounds_num = 500
	learners_num_list = [10, 100, 1000]
	participation_rates_list = [1, 0.5, 0.1]
	initialization_states_list = [Model.InitializationStates.RANDOM,
								  Model.InitializationStates.BURNIN_MEAN_CONSENSUS,
								  Model.InitializationStates.BURNIN_SINGLETON,
								  Model.InitializationStates.ROUND_ROBIN]
	for learners_num in learners_num_list:
		for participation_rate in participation_rates_list:
			for initialization_state in initialization_states_list:

				burnin_period = 0
				burnin_period_round_robin = 0
				if initialization_state == Model.InitializationStates.BURNIN_SINGLETON \
						or initialization_state == Model.InitializationStates.BURNIN_MEAN_CONSENSUS:
					burnin_period = 50
				if initialization_state == Model.InitializationStates.ROUND_ROBIN:
					burnin_period_round_robin = 1

				federated_training = ModelTraining.FederatedTraining(learners_num=learners_num,
																	 rounds_num=rounds_num,
																	 participation_rate=participation_rate,
																	 local_epochs=4,
																	 batch_size=32,
																	 initialization_state=initialization_state,
																	 burnin_period_epochs=burnin_period)
				print(federated_training.execution_stats)

				idx = list(range(len(x_train)))
				# TODO Shuffle the dataset for non-IID - sentences will not be loaded sequentially.
				# random.shuffle(idx)
				x_train = x_train[idx]
				y_train = y_train[idx]


				""" Create learners data distribution. """
				chunk_size = int(len(x_train) / learners_num)
				x_chunk, y_chunk = [], []
				for i in range(learners_num):
					x_chunk.append(x_train[idx[i * chunk_size:(i + 1) * chunk_size]])
					y_chunk.append(y_train[idx[i * chunk_size:(i + 1) * chunk_size]])
				x_chunk = np.array(x_chunk)
				y_chunk = np.array(y_chunk)
				print(f'Chunk size {chunk_size}', x_chunk.shape, y_chunk.shape)

				federated_training_results = federated_training.start(get_model_fn=model, x_train_chunks=x_chunk,
																	  y_train_chunks=y_chunk, x_test=x_test,
																	  y_test=y_test, info="IMDB")
				print(federated_training.execution_stats)

				output_filename = output_filename_template.format(rounds_num,
																  learners_num,
																  str(participation_rate).replace(".", ""),
																  initialization_state,
																  burnin_period)

				with open(output_filename, "w+", encoding='utf-8') as fout:
					json.dump(federated_training.execution_stats, fout, ensure_ascii=False, indent=4)


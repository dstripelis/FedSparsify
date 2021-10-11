from simulatedFL.models.model import Model
from simulatedFL.models.cifar10_cnn import Cifar10CNN
from simulatedFL.utils.model_training import ModelTraining

import os
import json
import random
import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
np.random.seed(1990)
random.seed(1990)
tf.random.set_seed(1990)

if __name__ == "__main__":

	"""Model Definition."""
	model = Cifar10CNN(kernel_initializer=Model.KERAS_INITIALIZER_GLOROT_UNIFORM).get_model
	model().summary()

	"""Load the data."""
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

	x_train = (x_train.astype('float32') / 256).reshape(-1, 32, 32, 3)
	x_test = (x_test.astype('float32') / 256).reshape(-1, 32, 32, 3)

	output_filename_template = "fedinit_states/logs/Cifar10/Cifar10.rounds_{}.learners_{}.participation_{}.init_{}.burnin_{}.json"
	rounds_num = 500
	learners_num_list = [10, 100, 1000]
	participation_rates_list = [1, 0.5, 0.1]
	initialization_states_list = [ModelTraining.FederatedTraining.INITIALIZATION_STATE_RANDOM,
								  ModelTraining.FederatedTraining.INITIALIZATION_STATE_BURNIN_MEAN_CONSENSUS,
								  ModelTraining.FederatedTraining.INITIALIZATION_STATE_BURNIN_SINGLETON,
								  ModelTraining.FederatedTraining.INITIALIZATION_STATE_ROUND_ROBIN]
	for learners_num in learners_num_list:
		for participation_rate in participation_rates_list:
			for initialization_state in initialization_states_list:

				burnin_period = 0
				burnin_period_round_robin = 0
				if initialization_state == ModelTraining.FederatedTraining.INITIALIZATION_STATE_BURNIN_SINGLETON \
						or initialization_state == ModelTraining.FederatedTraining.INITIALIZATION_STATE_BURNIN_MEAN_CONSENSUS:
					burnin_period = 50
				if initialization_state == ModelTraining.FederatedTraining.INITIALIZATION_STATE_ROUND_ROBIN:
					burnin_period_round_robin = 1

				federated_training = ModelTraining.FederatedTraining(learners_num=learners_num,
																	 rounds_num=rounds_num,
																	 participation_rate=participation_rate,
																	 local_epochs=4,
																	 batch_size=32,
																	 initialization_state=initialization_state,
																	 burnin_period_epochs=burnin_period,
																	 round_robin_period_epochs=burnin_period_round_robin)
				print(federated_training.execution_stats)

				"""Shuffle the dataset."""
				idx = list(range(len(x_train)))
				random.shuffle(idx)
				x_train = x_train[idx]
				# TODO sort dataset by yaxis and rerun for non-IID.
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
																	  y_test=y_test, info="CIFAR-10")
				print(federated_training.execution_stats)

				output_filename = output_filename_template.format(rounds_num,
																  learners_num,
																  str(participation_rate).replace(".", ""),
																  initialization_state,
																  burnin_period)

				with open(output_filename, "w+", encoding='utf-8') as fout:
					json.dump(federated_training.execution_stats, fout, ensure_ascii=False, indent=4)


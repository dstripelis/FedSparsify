from simulatedFL.models.model import Model
from simulatedFL.models.fashion_mnist_fc import FashionMnistModel
from simulatedFL.utils.model_training import ModelTraining
from simulatedFL.utils.data_distribution import PartitioningScheme

import os
import json
import random
import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
np.random.seed(1990)
random.seed(1990)
tf.random.set_seed(1990)

if __name__ == "__main__":

	"""Model Definition."""
	model = FashionMnistModel(kernel_initializer=Model.InitializationStates.GLOROT_UNIFORM).get_model
	model().summary()

	"""Load the data."""
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
	x_train = (x_train.astype('float32') / 256).reshape(-1, 28, 28, 1)
	x_test = (x_test.astype('float32') / 256).reshape(-1, 28, 28, 1)

	# centralized_training = ModelTraining.CentralizedTraining(epochs=300, batch_size=32)
	# centralized_training_results = centralized_training.start(model, x_train, y_train, x_test, y_test,
	# 														  info="Fashion-MNIST Centralized")
	# output_filename = "fedinit_states/logs/FashionMNIST/centralized.json"
	# with open(output_filename, "w+", encoding='utf-8') as fout:
	# 	json.dump(centralized_training_results.execution_stats, fout, ensure_ascii=False, indent=4)

	output_filename_template = \
		"fedinit_states/logs/FashionMNIST/FashionMNIST.rounds_{}.learners_{}.participation_{}.init_{}.burnin_{}.json"
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
																	 burnin_period_epochs=burnin_period,
																	 round_robin_period_epochs=burnin_period_round_robin)
				print(federated_training.execution_stats)
				pscheme = PartitioningScheme(x_train=x_train, y_train=y_train, partitions_num=learners_num)
				x_chunk, y_chunk = pscheme.non_iid_partition(classes_per_partition=2)

				federated_training_results = federated_training.start(get_model_fn=model, x_train_chunks=x_chunk,
																	  y_train_chunks=y_chunk, x_test=x_test,
																	  y_test=y_test, info="Fashion-MNIST")
				print(federated_training.execution_stats)

				output_filename = output_filename_template.format(rounds_num,
																  learners_num,
																  str(participation_rate).replace(".", ""),
																  initialization_state,
																  burnin_period)

				with open(output_filename, "w+", encoding='utf-8') as fout:
					json.dump(federated_training.execution_stats, fout, ensure_ascii=False, indent=4)


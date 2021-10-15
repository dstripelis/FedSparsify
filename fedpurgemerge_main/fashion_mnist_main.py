from simulatedFL.models.model import Model
from simulatedFL.models.fashion_mnist_fc import FashionMnistModel
from simulatedFL.utils.model_training import ModelTraining
from simulatedFL.utils.data_distribution import PartitioningScheme
from simulatedFL.utils.model_merge import MergeWeightedAverage
from simulatedFL.utils.model_purge import PurgeByThreshold

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

	""" Model Definition. """
	model = FashionMnistModel(kernel_initializer=Model.InitializationStates.GLOROT_UNIFORM).get_model
	model().summary()

	""" Load the data. """
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
	x_train = (x_train.astype('float32') / 256).reshape(-1, 28, 28, 1)
	x_test = (x_test.astype('float32') / 256).reshape(-1, 28, 28, 1)

	output_filename_template = os.path.dirname(__file__) + \
		"/../logs/FashionMNIST/FashionMNIST.rounds_{}.learners_{}.participation_{}.compression_{}.finetuning_{}.json"
	rounds_num = 90
	learners_num_list = [1]
	participation_rates_list = [1]
	compression_fractions = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
	local_epochs = 1
	fine_tuning_epochs = [0, 1, 2, 5, 10]

	for learners_num in learners_num_list:
		for participation_rate in participation_rates_list:
			for compression_fraction in compression_fractions:
				for fine_tuning_epoch_num in fine_tuning_epochs:
					pscheme = PartitioningScheme(x_train=x_train, y_train=y_train, partitions_num=learners_num)
					x_chunks, y_chunks = pscheme.iid_partition()
					# x_chunks, y_chunks = pscheme.non_iid_partition(classes_per_partition=10)

					scaling_factors = [y_chunk.size for y_chunk in y_chunks]
					merge_op = MergeWeightedAverage(scaling_factors)
					purge_op = PurgeByThreshold(compression_fraction=compression_fraction)
					federated_training = ModelTraining.FederatedTraining(merge_op=merge_op,
																		 learners_num=learners_num,
																		 rounds_num=rounds_num,
																		 local_epochs=local_epochs,
																		 learners_scaling_factors=scaling_factors,
																		 participation_rate=participation_rate,
																		 batch_size=32,
																		 purge_op=purge_op,
																		 fine_tuning_epochs=fine_tuning_epoch_num)
					federated_training.execution_stats['federated_environment']['compression_fraction'] = compression_fraction
					print(federated_training.execution_stats)
					federated_training_results = federated_training.start(get_model_fn=model, x_train_chunks=x_chunks,
																		  y_train_chunks=y_chunks, x_test=x_test,
																		  y_test=y_test, info="Fashion-MNIST")

					output_filename = output_filename_template.format(rounds_num,
																	  learners_num,
																	  str(participation_rate).replace(".", ""),
																	  str(compression_fraction).replace(".", ""),
																	  fine_tuning_epoch_num)

					with open(output_filename, "w+", encoding='utf-8') as fout:
						json.dump(federated_training_results, fout, ensure_ascii=False, indent=4)

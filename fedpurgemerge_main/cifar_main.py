from simulatedFL.models.model import Model
from simulatedFL.models.cifar_resnet50 import CifarResNet50
from simulatedFL.utils.model_training import ModelTraining
from simulatedFL.utils.data_distribution import PartitioningScheme
from simulatedFL.utils.model_merge import MergeWeightedAverage
from simulatedFL.utils.model_purge import PurgeByWeightMagnitude

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

	num_classes = 100
	if num_classes == 100:
		"""Model Definition."""
		model = CifarResNet50(classes_num=100).get_model
		model().summary()

		"""Load the data."""
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
		output_logs_dir = os.path.dirname(__file__) + "/../logs/Cifar100/"
		output_npzarrays_dir = os.path.dirname(__file__) + "/../npzarrays/Cifar100/"
		experiment_template = "Cifar100.rounds_{}.learners_{}.participation_{}.le_{}.compression_{}.finetuning_{}"
	else:
		"""Model Definition."""
		model = CifarResNet50(classes_num=10).get_model
		model().summary()

		"""Load the data."""
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
		output_logs_dir = os.path.dirname(__file__) + "/../logs/Cifar10/"
		output_npzarrays_dir = os.path.dirname(__file__) + "/../npzarrays/Cifar10/"
		experiment_template = "Cifar10.rounds_{}.learners_{}.participation_{}.le_{}.compression_{}.finetuning_{}"


	x_train = (x_train.astype('float32') / 256).reshape(-1, 32, 32, 3)
	x_test = (x_test.astype('float32') / 256).reshape(-1, 32, 32, 3)

	rounds_num = 100
	learners_num_list = [10]
	# participation_rates_list = [1, 0.5, 0.1]
	participation_rates_list = [1]
	sparsity_levels = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
	# sparsity_levels = [0.1, 0.2, 0.3, 0.4]
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
																		  y_test=y_test, info="Cifar")

					execution_output_filename = output_logs_dir + filled_in_template + ".json"
					with open(execution_output_filename, "w+", encoding='utf-8') as fout:
						json.dump(federated_training_results, fout, ensure_ascii=False, indent=4)

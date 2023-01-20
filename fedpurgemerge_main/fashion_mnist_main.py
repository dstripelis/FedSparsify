from simulatedFL.models.model import Model
from simulatedFL.models.fashion_mnist_fc import FashionMnistModel
from simulatedFL.utils.model_state import ModelState
from simulatedFL.utils.model_training import ModelTraining
from simulatedFL.utils.data_distribution import PartitioningScheme
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1

import os
import json
import random
import numpy as np
import tensorflow as tf

import simulatedFL.utils.model_merge as merge_ops
import simulatedFL.utils.model_purge as purge_ops

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
np.random.seed(1990)
random.seed(1990)
tf.random.set_seed(1990)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":

	""" Model Definition. """
	lambda1 = l1(0.0001)
	lambda2 = l2(0.0001)

	model = FashionMnistModel(kernel_initializer=Model.InitializationStates.GLOROT_UNIFORM, learning_rate=0.02,
							  use_sgd=True, use_fedprox=False, use_sgd_with_momentum=False, fedprox_mu=0.0,
							  momentum_factor=0.0, kernel_regularizer=None, bias_regularizer=None).get_model
	model().summary()

	""" Load the data. """
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
	x_train = (x_train.astype('float32') / 256).reshape(-1, 28, 28, 1)
	x_test = (x_test.astype('float32') / 256).reshape(-1, 28, 28, 1)

	output_logs_dir = os.path.dirname(__file__) + "/../logs/FashionMNIST/test/"
	output_npzarrays_dir = os.path.dirname(__file__) + "/../npzarrays/FashionMNIST/"
	experiment_template = \
		"FashionMNIST.ExponentExploration_exp{}.NonIID.rounds_{}.learners_{}.participation_{}.le_{}.compression_{}.sparsificationround_{}.sparsifyevery_{}rounds.finetuning_{}"

	rounds_num = 200
	learners_num_list = [10]
	participation_rates_list = [1]

	# One-Shot Pruning
	# sparsity_levels = [0.8, 0.85, 0.9, 0.95, 0.99]
	# sparsification_frequency = [1]
	# start_sparsification_at_round = [190]

	# Federated Progressive Pruning
	exponents = [1, 6, 12]
	sparsity_levels = [0.95]
	sparsification_frequency = [1, 2]
	start_sparsification_at_round = [1]

	local_epochs = 4
	fine_tuning_epochs = [0]
	batch_size = 32
	train_with_global_mask = True

	for learners_num, participation_rate  in zip(learners_num_list, participation_rates_list):
		for exponent in exponents:
			for sparsity_level in sparsity_levels:
				for frequency in sparsification_frequency:
					for sparsification_round in start_sparsification_at_round:
						for fine_tuning_epoch_num in fine_tuning_epochs:

							# fill in string placeholders
							filled_in_template = experiment_template.format(int(exponent),
																			rounds_num,
																			learners_num,
																			str(participation_rate).replace(".", ""),
																			str(local_epochs),
																			str(sparsity_level).replace(".", ""),
																			str(sparsification_round),
																			str(frequency),
																			fine_tuning_epoch_num)
							output_arrays_dir = output_npzarrays_dir + filled_in_template

							pscheme = PartitioningScheme(x_train=x_train, y_train=y_train, partitions_num=learners_num)
							# x_chunks, y_chunks = pscheme.iid_partition()
							x_chunks, y_chunks = pscheme.non_iid_partition(classes_per_partition=2)

							scaling_factors = [y_chunk.size for y_chunk in y_chunks]

							# Merging Ops.
							merge_op = merge_ops.MergeWeightedAverage(scaling_factors)
							# merge_op = merge_ops.MergeMedian(scaling_factors)
							# merge_op = merge_ops.MergeAbsMax(scaling_factors)
							# merge_op = merge_ops.MergeAbsMin(scaling_factors, discard_zeroes=True)
							# merge_op = merge_ops.MergeTanh(scaling_factors)
							# merge_op = merge_ops.MergeWeightedAverageNNZ(scaling_factors)
							# merge_op = merge_ops.MergeWeightedAverageMajorityVoting(scaling_factors)
							# merge_op = merge_ops.MergeWeightedPseudoGradients(scaling_factors)

							# Purging Ops.
							# purge_op = purge_ops.PurgeByWeightMagnitude(sparsity_level=sparsity_level)
							# purge_op = purge_ops.PurgeByNNZWeightMagnitude(sparsity_level=sparsity_level,
							# 											   sparsify_every_k_round=frequency)
							# purge_op = purge_ops.PurgeByNNZWeightMagnitudeRandom(sparsity_level=sparsity_level,
							# 													 num_params=model().count_params(),
							# 													 sparsify_every_k_round=frequency)
							# purge_op = purge_ops.PurgeByLayerWeightMagnitude(sparsity_level=sparsity_level)
							# purge_op = purge_ops.PurgeByLayerNNZWeightMagnitude(sparsity_level=sparsity_level)
							# purge_op = purge_ops.PurgeByWeightMagnitudeRandomGradual(model=model(),
							# 														 start_at_round=sparsification_round,
							# 														 sparsity_level_init=0.0,
							# 														 sparsity_level_final=sparsity_level,
							# 														 total_rounds=rounds_num,
							# 														 delta_round_pruning=frequency,
							# 														 exponent=3,
							# 														 federated_model=True)
							purge_op = purge_ops.PurgeByWeightMagnitudeGradual(start_at_round=sparsification_round,
																			   sparsity_level_init=0.0,
																			   sparsity_level_final=sparsity_level,
																			   total_rounds=rounds_num,
																			   delta_round_pruning=frequency,
																			   exponent=exponent,
																			   purge_per_layer=False,
																			   federated_model=True)

							# OneShot
							# purge_op = purge_ops.PurgeByWeightMagnitudeGradual(start_at_round=sparsification_round,
							# 												   sparsity_level_init=sparsity_level,
							# 												   sparsity_level_final=sparsity_level,
							# 												   total_rounds=rounds_num,
							# 												   delta_round_pruning=frequency,
							# 												   exponent=3,
							# 												   purge_per_layer=False,
							# 												   federated_model=True)

							# Random Learner
							# randint = random.randint(0, learners_num-1)
							# purge_op = purge_ops.PurgeSNIP(model(),
							# 							   sparsity=sparsity_level,
							# 							   x=x_chunks[randint][:batch_size],
							# 							   y=y_chunks[randint][:batch_size])

							# All-Learners
							# learners_masks = []
							# for lidx in range(learners_num):
							# 	learner_masks = purge_ops.PurgeSNIP(model(),
							# 								   sparsity=sparsity_level,
							# 								   x=x_chunks[lidx][:batch_size],
							# 								   y=y_chunks[lidx][:batch_size])
							# 	learners_masks.append(learner_masks.precomputed_masks)
							# masks_sum = learners_masks[0]
							# for model_masks in learners_masks[1:]:
							# 	masks_sum = [np.add(m1.flatten(), m2.flatten()) for m1, m2 in zip(masks_sum, model_masks)]
							# masks_majority_voting = []
							# for midx, m in enumerate(masks_sum):
							# 	binary_mask = np.array([1 if p > 0 else 0 for p in m])
							# 	binary_mask = binary_mask.reshape(model().get_weights()[midx].shape)
							# 	masks_majority_voting.append(binary_mask)

							# Random Learner
							# randint = random.randint(0, learners_num-1)
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
																				 purge_op_global=purge_op,
																				 start_purging_at_round=sparsification_round,
																				 fine_tuning_epochs=fine_tuning_epoch_num,
																				 train_with_global_mask=train_with_global_mask,
																				 start_training_with_global_mask_at_round=sparsification_round,
																				 output_arrays_dir=output_arrays_dir)
																				 # precomputed_masks=masks_majority_voting)
																				 # precomputed_masks=purge_op.precomputed_masks)
							federated_training.execution_stats['federated_environment']['model_params'] = ModelState.count_non_zero_elems(model())
							federated_training.execution_stats['federated_environment']['sparsity_level'] = sparsity_level
							federated_training.execution_stats['federated_environment']['additional_specs'] = purge_op.json()
							federated_training.execution_stats['federated_environment']['data_distribution'] = \
								pscheme.to_json_representation()
							federated_training_results = federated_training.start(get_model_fn=model, x_train_chunks=x_chunks,
																				  y_train_chunks=y_chunks, x_test=x_test,
																				  y_test=y_test, info="Fashion-MNIST")

							execution_output_filename = output_logs_dir + filled_in_template + ".json"
							with open(execution_output_filename, "w+", encoding='utf-8') as fout:
								json.dump(federated_training_results, fout, ensure_ascii=False, indent=4)

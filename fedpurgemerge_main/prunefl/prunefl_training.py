import json
import os
import random
import time

import numpy as np
import tensorflow as tf

from collections import namedtuple
from simulatedFL.models.model import Model
from simulatedFL.utils.model_merge import MergeOps
from simulatedFL.utils.model_purge import PurgeOps
from simulatedFL.utils.masked_callback import MaskedCallback
from simulatedFL.utils.logger import CustomLogger
from simulatedFL.utils.model_state import ModelState

FederationRoundMetadata = namedtuple(typename='FederationRoundMetadata', field_names=['round_id', 'processing_time',
																					  'global_model_total_params',
																					  'global_model_test_loss',
																					  'global_model_test_score',
																					  'local_models_total_params_after_training',
																					  'local_models_total_params_after_purging',
																					  'local_models_train_loss',
																					  'local_models_train_score',
																					  'local_models_test_loss',
																					  'local_models_test_score'])

class PruneFLTraining:

	class PruneFLReadjustment(object):

		def __init__(self, trainable_vars_times, non_trainable_vars_times, prune_rate=0.3):
			self.trainable_vars_times = trainable_vars_times
			self.non_trainable_vars_times = non_trainable_vars_times
			self.prune_rate = prune_rate

		@classmethod
		def compute_gradients(cls, model, x, y):
			x = tf.convert_to_tensor(x)
			y = tf.convert_to_tensor(y)

			with tf.GradientTape() as tape:
				y_pred = model(x)
				loss = model.compiled_loss(y, y_pred)

			# Gradients are only computed for the trainable weights/variables.
			grads = tape.gradient(loss, [v for v in model.trainable_variables])
			grads_np = []
			for g in grads:
				if isinstance(g, tf.IndexedSlices):
					dense_shape = g.dense_shape.numpy()
					g = g.values.numpy().flatten()[:np.prod(dense_shape)]
					g = g.reshape(dense_shape)
					grads_np.append(g)
				else:
					grads_np.append(np.array(g))
			return grads_np

		def readjust_model_masks(self, model, gradients, federation_round=0, total_rounds=200):
			num_remaining_params = lambda r, R: 1 - self.prune_rate * np.power(0.5, r / R)
			importances = []
			for i, g in enumerate(gradients):
				g = np.square(g)
				g = np.divide(g, self.trainable_vars_times[i])
				importances.append(g)

			t = 0.2
			delta_M = 0
			cat_grad = np.concatenate([g.flatten() for g in gradients])
			cat_imp = np.concatenate([i.flatten() for i in importances])
			# We need descending order, hence the negation.
			indices = np.argsort(-cat_grad)
			n_required = num_remaining_params(federation_round, total_rounds) * cat_grad.size
			n_grown = 0

			masks = []
			for g in gradients:
				masks.append(np.zeros_like(g))

			for j, i in enumerate(indices):
				if cat_imp[i] >= delta_M / t or n_grown <= n_required:
					index_within_layer = i.item()
					for layer_idx in range(len(self.trainable_vars_times)):
						size = gradients[layer_idx].size
						if index_within_layer >= size:
							index_within_layer -= size
						else:
							break

					delta_M += cat_grad[i]
					t += self.trainable_vars_times[layer_idx]

					shape = tuple(masks[layer_idx].shape)
					masks[layer_idx][np.unravel_index(index_within_layer, shape)] = 1
					n_grown += 1
				else:
					break

			print("Readjustment Density:", n_grown / cat_imp.size)

			# Set up the masks by reconstructing the sequence of the
			# model's trainable and non-trainable parameters.
			final_model_masks = []
			trainable_var_masks_iter = iter(masks)
			for v in model.variables:
				if v.trainable and 'batch_norm' not in v.name:
					final_model_masks.append(next(trainable_var_masks_iter))
				else:
					final_model_masks.append(np.ones(v.shape))
			return final_model_masks


	class FederatedTraining:

		def __init__(self, merge_op, learners_num, rounds_num, local_epochs, learners_scaling_factors,
					 participation_rate=1, batch_size=100, purge_op_local=None, purge_op_global=None,
					 start_purging_at_round=0, fine_tuning_epochs=0, train_with_global_mask=False,
					 start_training_with_global_mask_at_round=0, initialization_state=Model.InitializationStates.RANDOM,
					 burnin_period_epochs=0, round_robin_period_epochs=0, output_arrays_dir=None, precomputed_masks=None,
					 masks_readjustment_rounds=50, prunefl_training_context=None):

			# Required attributes.
			assert isinstance(merge_op, MergeOps)
			self.merge_op = merge_op
			self.learners_num = learners_num
			self.rounds_num = rounds_num
			self.local_epochs = local_epochs
			self.learners_scaling_factors = learners_scaling_factors

			# Optional attributes.
			self.participation_rate = participation_rate
			self.batch_size = batch_size

			# Optional attributes related to federated purging and merging.
			self.purge_op_local = purge_op_local
			self.purge_op_global = purge_op_global
			self.start_purging_at_round = start_purging_at_round
			self.fine_tuning_epochs = fine_tuning_epochs

			# Whether training is carried out exclusively on the parameters of the global model.
			self.train_with_global_mask = train_with_global_mask
			self.start_training_with_global_mask_at_round = start_training_with_global_mask_at_round

			# Optional attributes related to federated model initialization state experiments.
			self.initialization_state = initialization_state
			self.burnin_period_epochs = burnin_period_epochs
			self.round_robin_period_epochs = round_robin_period_epochs

			# Create the directory if not exists.
			self.output_arrays_dir = output_arrays_dir
			if not os.path.exists(self.output_arrays_dir):
				os.makedirs(self.output_arrays_dir)
			self.local_model_fname_template = os.path.join(self.output_arrays_dir,
														   "local_model_{}_federation_round_{}.npz")
			self.local_model_masks_fname_template = os.path.join(self.output_arrays_dir,
																 "local_model_{}_masks_federation_round_{}.npz")
			self.global_model_fname_template = os.path.join(self.output_arrays_dir,
															"global_model_federation_round_{}.npz")
			self.global_model_masks_fname_template = os.path.join(self.output_arrays_dir,
																  "global_model_masks_federation_round_{}.npz")

			self.precomputed_masks = precomputed_masks
			self.masks_readjustment_rounds = masks_readjustment_rounds
			self.prunefl_training_context = prunefl_training_context

			# A json representation of the federated session with its associated training results.
			self.execution_stats = self._construct_federated_execution_stats()

			# Print the execution stats.
			CustomLogger.info(self.execution_stats)

		def _construct_federated_execution_stats(self):
			res = {
				"federated_environment" : {
					"merge_function": None if self.merge_op is None else str(self.merge_op.__class__.__name__),
					"number_of_learners" : self.learners_num,
					"federation_rounds": self.rounds_num,
					"local_epochs_per_client": self.local_epochs,
					"scaling_factors": self.learners_scaling_factors,
					"participation_rate": self.participation_rate,
					"batch_size_per_client" : self.batch_size,
					"purge_function_local": None if self.purge_op_local is None else str(self.purge_op_local.__class__.__name__),
					"purge_function_global": None if self.purge_op_global is None else str(self.purge_op_global.__class__.__name__),
					"start_purging_at_round": self.start_purging_at_round,
					"train_with_global_mask": self.train_with_global_mask,
					"start_training_with_global_mask_at_round": self.start_training_with_global_mask_at_round,
					"fine_tuning_epochs": self.fine_tuning_epochs,
					"federated_model_initialization_state" : self.initialization_state,
					"burnin_period_epochs" : self.burnin_period_epochs,
					"round_robin_period_epochs": self.round_robin_period_epochs,
					"with_precomputed_masks": True if self.precomputed_masks is not None else False,
					"masks_readjustment_rounds": self.masks_readjustment_rounds
				},
				"federated_execution_results" : list()
			}
			return res


		def set_initial_federation_model_state(self, gmodel, lmodels, x_train_chunks, y_train_chunks):

			model_state = gmodel.get_weights()
			if self.initialization_state == Model.InitializationStates.RANDOM:
				# random initialization - Assign the random state to every learner
				model_state = gmodel.get_weights()

			elif self.initialization_state == Model.InitializationStates.BURNIN_SINGLETON:
				# burnin initialization - 'Burn' a single learner and assign its weight
				lmodels[0].fit(x_train_chunks[0], y_train_chunks[0],
					epochs=self.burnin_period_epochs, batch_size=self.batch_size, verbose=False
				)
				model_state = lmodels[0].get_weights()

			elif self.initialization_state == Model.InitializationStates.BURNIN_MEAN_CONSENSUS:
				# burnin initialization - 'Burn' all learners - average them and assign their weight
				for lidx, lmodel in enumerate(lmodels):
					lmodels[lidx].fit(x_train_chunks[lidx], y_train_chunks[lidx],
						epochs=self.burnin_period_epochs, batch_size=self.batch_size, verbose=False
					)
				num_weights = len(gmodel.get_weights())
				init_state = []
				for j in range(num_weights):
					learners_weights = [model.get_weights()[j] for model in lmodels]
					learners_weights_np = np.array(learners_weights)
					mean_weight = learners_weights_np.mean(axis=0)
					init_state.append(mean_weight)
				model_state = init_state

			elif self.initialization_state == Model.InitializationStates.BURNIN_SCALED_CONSENSUS:
				# burnin initialization - 'Burn' all learners and scale them
				for lidx, lmodel in enumerate(lmodels):
					lmodel.fit(x_train_chunks[lidx], y_train_chunks[lidx],
						epochs=self.burnin_period_epochs, batch_size=self.batch_size, verbose=False
					)
				num_weights = len(gmodel.get_weights())
				init_state = []
				contrib_value = np.sqrt(np.divide(1, len(lmodels)))
				for j in range(num_weights):
					learners_weights = [contrib_value * model.get_weights()[j] for model in lmodels]
					learners_weights_np = np.array(learners_weights)
					mean_weight = learners_weights_np.sum(axis=0)
					init_state.append(mean_weight)
				model_state = init_state

			elif self.initialization_state == Model.InitializationStates.ROUND_ROBIN_RATE_SAMPLE:
				random_models_idx = self.generate_random_participating_learners()
				model_state = lmodels[0].get_weights()
				for lidx, lmodel in enumerate(lmodels):
					# We perform the round-robin initialization based on the participation rate.
					# If the rate is 1, then consider all learners during the initialization step, else
					# we loop over a pool of learners (sample).
					if lidx in random_models_idx:
						lmodel.set_weights(model_state)
						lmodel.fit(x_train_chunks[lidx], y_train_chunks[lidx],
							epochs=self.round_robin_period_epochs, batch_size=self.batch_size, verbose=False
						)
						model_state = lmodel.get_weights()

			return model_state


		def generate_random_participating_learners(self):
			random_models_idx = random.sample([x for x in range(self.learners_num)],
											  k=int(self.participation_rate * self.learners_num))
			return random_models_idx


		def train_models(self, models, epochs_num, x_train_chunks, y_train_chunks, models_masks=None):
			batch_size = self.batch_size
			if isinstance(x_train_chunks[0], tf.data.Dataset):
				batch_size = None

			for midx, model in enumerate(models):

				callbacks = []
				if models_masks is not None:
					"We always apply the mask (if it is given), even when a callback will not take place."
					masks = models_masks[midx]
					PurgeOps.apply_model_masks(model, masks)
					callbacks.append(MaskedCallback(model_masks=masks))

				model.fit(
					x_train_chunks[midx], y_train_chunks[midx],
					epochs=epochs_num, batch_size=batch_size, verbose=False, callbacks=callbacks
				)
			return models


		def evaluate_models(self, models, x_train_chunks, y_train_chunks, x_test, y_test, models_masks=None):
			models_train_loss = []
			models_train_score = []
			models_test_loss = []
			models_test_score = []
			for midx, model in enumerate(models):

				callbacks = []
				if models_masks is not None:
					masks = models_masks[midx]
					callbacks.append(MaskedCallback(model_masks=masks))

				# evaluate models on local dataset
				model_train_loss, model_train_score = model.evaluate(x_train_chunks[midx], y_train_chunks[midx],
																		verbose=False, callbacks=callbacks)
				models_train_loss.append(model_train_loss)
				models_train_score.append(model_train_score)

				# evaluate models og test dataset
				model_test_loss, model_test_score = model.evaluate(x_test, y_test, verbose=False, callbacks=callbacks)
				models_test_loss.append(model_test_loss)
				models_test_score.append(model_test_score)

			return models_train_loss, models_train_score, models_test_loss, models_test_score


		def start(self, get_model_fn,
				  x_train_chunks_as_numpy,
				  y_train_chunks_as_numpy,
				  x_train_chunks_as_datasets,
				  y_train_chunks_as_datasets,
				  x_test, y_test, info="Some Feedback"):
			"""Federated Training."""

			lmodels = [get_model_fn() for _ in range(self.learners_num)]
			gmodel = get_model_fn()
			if self.precomputed_masks is not None:
				PurgeOps.apply_model_masks(gmodel, self.precomputed_masks)
			CustomLogger.info(("{}".format(info)))

			federation_rounds_stats = []
			global_weights = self.set_initial_federation_model_state(
				gmodel, lmodels, x_train_chunks_as_datasets, y_train_chunks_as_datasets)
			gmodel.set_weights(global_weights)
			gmodel_binary_masks = ModelState.get_model_binary_masks(gmodel)
			# count number of model parameters
			global_model_num_params = ModelState.count_non_zero_elems(gmodel)

			# eval global model
			gmodel_loss, gmodel_score = gmodel.evaluate(x_test, y_test, verbose=False)

			# metadata collectors
			non_zero_elements_after_training = []
			non_zero_elements_after_purging = []
			local_models_train_loss = []
			local_models_train_score = []
			local_models_test_loss = []
			local_models_test_score = []

			# register/log 0th round execution matadata - it is just the model performance of the initial state.
			federation_round_metadata = FederationRoundMetadata(round_id=0, processing_time=[],
																global_model_total_params=global_model_num_params,
																global_model_test_loss=gmodel_loss,
																global_model_test_score=gmodel_score,
																local_models_total_params_after_training=non_zero_elements_after_training,
																local_models_total_params_after_purging=non_zero_elements_after_purging,
																local_models_train_loss=local_models_train_loss,
																local_models_train_score=local_models_train_score,
																local_models_test_loss=local_models_test_loss,
																local_models_test_score=local_models_test_score)
			federation_rounds_stats.append(federation_round_metadata._asdict())

			for round_id in range(1, self.rounds_num + 1):

				# capture end time
				st = time.process_time()

				# set all local models to global model state
				for model in lmodels:
					model.set_weights(global_weights)

				# randomly select local models
				random_models_idx = self.generate_random_participating_learners()
				models_subset = [lmodels[idx] for idx in random_models_idx]
				x_train_chunks_subset = [x_train_chunks_as_datasets[idx] for idx in random_models_idx]
				y_train_chunks_subset = [y_train_chunks_as_datasets[idx] for idx in random_models_idx]
				scaling_factors_subset = [self.learners_scaling_factors[idx] for idx in random_models_idx]

				# print number of learners
				CustomLogger.info("Total Local Models: {}".format(len(models_subset)))

				# normal training w/ or w/o global mask - if True, we need to augment the global_model_masks collection
				# with one global mask per local model, hence the [gmodel_binary_masks] * len(models_subset) multiplication!
				global_model_masks = None
				if self.train_with_global_mask is True and round_id > self.start_training_with_global_mask_at_round:
					global_model_masks = [gmodel_binary_masks] * len(models_subset)

					# PruneFL logic!!!
					if round_id % self.masks_readjustment_rounds == 0:
						all_gradients = None
						for lmodel, x_chunk, y_chunk in \
								zip(models_subset, x_train_chunks_as_numpy, y_train_chunks_as_numpy):
							model_gradients = self.prunefl_training_context \
								.compute_gradients(lmodel, x_chunk, y_chunk)
							if all_gradients is None:
								all_gradients = model_gradients
							else:
								all_gradients = [np.add(g1, g2) for g1, g2 in zip(all_gradients, model_gradients)]
						model_masks = self.prunefl_training_context.readjust_model_masks(
							gmodel, all_gradients, federation_round=round_id, total_rounds=self.rounds_num)
						global_model_masks = [model_masks] * len(models_subset)

				# Train models.
				models_subset = self.train_models(models_subset, self.local_epochs,
												  x_train_chunks_subset,
												  y_train_chunks_subset,
												  models_masks=global_model_masks)

				# collection with local model sizes after training / before pruning for the current round
				non_zero_elements_after_training = [ModelState.count_non_zero_elems(model) for model in models_subset]
				CustomLogger.info("Local models NNZ-params after training: {}".format(non_zero_elements_after_training))

				if self.purge_op_local is not None and round_id > self.start_purging_at_round:
					# purge function __call__() definition: (purging_model, global_model, federation_round)
					models_subset_masks = [self.purge_op_local(lmodel, gmodel, round_id) for lmodel in models_subset]

					# fine-tuning on pruned_models
					models_subset = self.train_models(models_subset, self.fine_tuning_epochs,
													  x_train_chunks_subset, y_train_chunks_subset,
													  models_masks=models_subset_masks)
				else:
					models_subset_masks = [ModelState.get_model_binary_masks(lmodel) for lmodel in models_subset]

				# save local models masks
				# for lidx, lmodel in enumerate(models_subset):
				# 	lmodel_binary_masks = ModelState.get_model_binary_masks(lmodel)
				# 	np.savez(self.local_model_fname_template.format(lidx, round_id), *lmodel.get_weights())
				# 	np.savez(self.local_model_masks_fname_template.format(lidx, round_id), *lmodel_binary_masks)

				# collection with local model sizes after pruning for the current round
				non_zero_elements_after_purging = [ModelState.count_non_zero_elems(model) for model in models_subset]
				CustomLogger.info("Local models NNZ-params after purging: {}".format(non_zero_elements_after_purging))

				# evaluation of local models on local training and global test set
				local_models_train_loss, local_models_train_score, local_models_test_loss, local_models_test_score = \
					self.evaluate_models(models_subset,
										 x_train_chunks_as_datasets,
										 y_train_chunks_as_datasets,
										 x_test,
										 y_test,
										 models_subset_masks)

				self.merge_op.set_scaling_factors(scaling_factors_subset)
				global_weights = self.merge_op(models_subset, models_subset_masks, global_model=gmodel)
				# set global model weights
				gmodel.set_weights(global_weights)

				if self.purge_op_global is not None and round_id > self.start_purging_at_round:
					# purge function __call__() definition: (purging_model, global_model, federation_round)
					global_model_masks = self.purge_op_global(gmodel, gmodel, round_id)
					PurgeOps.apply_model_masks(gmodel, global_model_masks)

				# count number of model parameters
				global_model_num_params = ModelState.count_non_zero_elems(gmodel)
				# find global model binary masks
				gmodel_binary_masks = ModelState.get_model_binary_masks(gmodel)
				# save global model and masks masks
				# np.savez(self.global_model_fname_template.format(round_id), *gmodel.get_weights())
				# np.savez(self.global_model_masks_fname_template.format(round_id), *gmodel_binary_masks)

				# eval global model
				gmodel_loss, gmodel_score = gmodel.evaluate(x_test, y_test, verbose=False)

				# capture end time
				et = time.process_time()

				# raw elapsed time
				elapsed_time = et - st
				# avg elapsed time -- assuming every learner was running in parallel
				avg_elapsed_time = np.divide(elapsed_time, len(models_subset))
				CustomLogger.info(
					("Federation Round: {}, Loss: {}, Score: {}, Parameters: {}".format(
						round_id, gmodel_loss, gmodel_score, global_model_num_params)))

				# register/log federation round execution matadata
				federation_round_metadata = FederationRoundMetadata(round_id=round_id, processing_time=avg_elapsed_time,
																	global_model_total_params=global_model_num_params,
																	global_model_test_loss=gmodel_loss,
																	global_model_test_score=gmodel_score,
																	local_models_total_params_after_training=non_zero_elements_after_training,
																	local_models_total_params_after_purging=non_zero_elements_after_purging,
																	local_models_train_loss=local_models_train_loss,
																	local_models_train_score=local_models_train_score,
																	local_models_test_loss=local_models_test_loss,
																	local_models_test_score=local_models_test_score)
				federation_rounds_stats.append(federation_round_metadata._asdict())

			# clear memory
			del lmodels
			del gmodel
			tf.keras.backend.clear_session()

			self.execution_stats["federated_execution_results"] = federation_rounds_stats
			return self.execution_stats
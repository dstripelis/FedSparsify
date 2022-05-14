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
from simulatedFL.utils.pseudogradients_callback import PseudoGradientsCallback
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

class ModelTraining:

	class CentralizedTraining:

		def __init__(self, epochs=100, batch_size=100, fine_tune_epochs=0, purging_function=None):
			# TODO Implement purging operation if needed!! For now we run everything using the FederatedTraining class.
			self.epochs = epochs
			self.batch_size = batch_size
			self.fine_tune_epochs = fine_tune_epochs
			self.purging_function = purging_function

		def _construct_centralized_execution_stats(self):
			res = {
				"centralized_environment" : {
					"total_epochs": self.epochs,
					"batch_size": self.batch_size,
					"fine_tune_epochs": self.fine_tune_epochs,
					"purge_function": self.purging_function,
					"start_purging_at_epoch": self.epochs
				},
				"centralized_execution_results" : list()
			}
			return res

		def _model_fit(self, epochs, model, x_train, x_test, y_train, y_test):
			train_losses, train_scores, test_losses, test_scores = [], [], [], []
			for i in range(epochs):
				train_results = model.fit(x_train, y_train, epochs=1, batch_size=self.batch_size, verbose=True)
				train_loss, train_score = model.evaluate(x_train, x_test, batch_size=self.batch_size)
				test_loss, test_score = model.evaluate(x_test, y_test, batch_size=self.batch_size)
				train_losses.append(train_loss)
				train_scores.append(train_score)
				test_losses.append(test_loss)
				test_scores.append(test_score)
			return train_losses, train_scores, test_losses, test_scores

		def start(self, get_model_fn, x_train, y_train, x_test, y_test, info="Some Feedback"):
			"""Train."""
			CustomLogger.info(("{}".format(info)))
			model = get_model_fn()
			train_losses, train_scores, test_losses, test_scores = \
				self._model_fit(self.epochs, model, x_train, x_test, y_train, y_test)
			eval_results_before_purging = \
				{"train_losses": train_losses, "train_score": train_scores,
				 "test_losses": test_losses, "test_scores": test_scores}

			# if self.purging_function is not None:
			# 	self.purging_function(model, model, self.epochs)

			train_losses, train_scores, test_losses, test_scores = \
				self._model_fit(self.fine_tune_epochs, model, x_train, x_test, y_train, y_test)
			eval_results_after_purging = \
				{"train_losses": train_losses, "train_score": train_scores,
				 "test_losses": test_losses, "test_scores": test_scores}

			execution_stats = self._construct_centralized_execution_stats()
			execution_stats["centralized_execution_results"] = {
				"eval_results_before_pruning": eval_results_before_purging,
				"eval_results_after_pruning": eval_results_after_purging,
			}
			return execution_stats


	class FederatedTraining:

		def __init__(self, merge_op, learners_num, rounds_num, local_epochs, learners_scaling_factors,
					 participation_rate=1, batch_size=100, purge_op_local=None, purge_op_global=None,
					 start_purging_at_round=0, fine_tuning_epochs=0, train_with_global_mask=False,
					 start_training_with_global_mask_at_round=0, initialization_state=Model.InitializationStates.RANDOM,
					 burnin_period_epochs=0, round_robin_period_epochs=0, output_arrays_dir=None, precomputed_masks=None):

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

			# A json representation of the federated session with its associated training results.
			self.execution_stats = self._construct_federated_execution_stats()

			# Print the execution results.
			CustomLogger.info(self.execution_stats)

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
					"round_robin_period_epochs": self.round_robin_period_epochs
				},
				"federated_execution_results" : list()
			}
			return res


		def set_initial_federation_model_state(self, gmodel, lmodels, x_train_chunks, y_train_chunks):

			batch_size = self.batch_size
			if isinstance(x_train_chunks[0], tf.data.Dataset):
				batch_size = None

			model_state = []
			if self.initialization_state == Model.InitializationStates.RANDOM:
				# random initialization - Assign the random state to every learner
				model_state = gmodel.get_weights()

			elif self.initialization_state == Model.InitializationStates.BURNIN_SINGLETON:
				# burnin initialization - 'Burn' a single learner and assign its weight
				lmodels[0].fit(x_train_chunks[0], y_train_chunks[0],
					epochs=self.burnin_period_epochs, batch_size=batch_size, verbose=False
				)
				model_state = lmodels[0].get_weights()

			elif self.initialization_state == Model.InitializationStates.BURNIN_MEAN_CONSENSUS:
				# burnin initialization - 'Burn' all learners - average them and assign their weight
				for lidx, lmodel in enumerate(lmodels):
					lmodels[lidx].fit(x_train_chunks[lidx], y_train_chunks[lidx],
						epochs=self.burnin_period_epochs, batch_size=batch_size, verbose=False
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
						epochs=self.burnin_period_epochs, batch_size=batch_size, verbose=False
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
							epochs=self.round_robin_period_epochs, batch_size=batch_size, verbose=False
						)
						model_state = lmodel.get_weights()

			return model_state


		def generate_random_participating_learners(self):
			random_models_idx = random.sample([x for x in range(self.learners_num)],
											  k=int(self.participation_rate * self.learners_num))
			return random_models_idx


		def train_models(self, models, epochs_num, x_train_chunks, y_train_chunks, global_model, models_masks=None):
			batch_size = self.batch_size
			if isinstance(x_train_chunks[0], tf.data.Dataset):
				batch_size = None

			models_pseudogradients = []
			for midx, model in enumerate(models):

				pseudogradients_cback_obj = PseudoGradientsCallback(global_model=global_model)
				current_model_callbacks = [pseudogradients_cback_obj]
				if models_masks is not None:
					"We always apply the mask when it is given."
					masks = models_masks[midx]
					PurgeOps.apply_model_masks(model, masks)
					global_masks_callback = MaskedCallback(model_masks=masks)
					current_model_callbacks.append(global_masks_callback)

				model.fit(x_train_chunks[midx], y_train_chunks[midx],
						  epochs=epochs_num, batch_size=batch_size,
						  verbose=False, callbacks=current_model_callbacks)
				models_pseudogradients.append(pseudogradients_cback_obj.pseudogradients)
			return models, models_pseudogradients


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


		def start(self, get_model_fn, x_train_chunks, y_train_chunks, x_test, y_test, info="Some Feedback"):
			"""Federated Training."""

			lmodels = [get_model_fn() for _ in range(self.learners_num)]
			gmodel = get_model_fn()
			if self.precomputed_masks is not None:
				PurgeOps.apply_model_masks(gmodel, self.precomputed_masks)
			CustomLogger.info(("{}".format(info)))

			federation_rounds_stats = []
			global_weights = self.set_initial_federation_model_state(gmodel, lmodels, x_train_chunks, y_train_chunks)
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
				x_train_chunks_subset = [x_train_chunks[idx] for idx in random_models_idx]
				y_train_chunks_subset = [y_train_chunks[idx] for idx in random_models_idx]
				scaling_factors_subset = [self.learners_scaling_factors[idx] for idx in random_models_idx]

				# print number of learners
				CustomLogger.info("Total Local Models: {}".format(len(models_subset)))

				# normal training w/ or w/o global mask - if True, we need to augment the global_model_masks collection
				# with one global mask per local model, hence the [gmodel_binary_masks] * len(models_subset) multiplication!
				global_model_masks_learners = None
				if self.precomputed_masks or \
						(self.train_with_global_mask is True and
						 round_id > self.start_training_with_global_mask_at_round):
					global_model_masks_learners = [gmodel_binary_masks] * len(models_subset)

				# Train models.
				models_subset, models_subset_pseudogradients = self.train_models(models_subset,
																				 self.local_epochs,
																				 x_train_chunks_subset,
																				 y_train_chunks_subset,
																				 global_model=gmodel,
																				 models_masks=global_model_masks_learners)

				# collection with local model sizes after training and before purging for the current round
				non_zero_elements_after_training = [ModelState.count_non_zero_elems(model) for model in models_subset]
				CustomLogger.info("Local models NNZ-params after training: {}".format(non_zero_elements_after_training))

				if self.purge_op_local is not None and round_id > self.start_purging_at_round:
					if self.train_with_global_mask:
						for lmodel in models_subset:
							PurgeOps.apply_model_masks(lmodel, gmodel_binary_masks)
					# purge function __call__() definition: (purging_model, global_model, federation_round)
					# get local models masks after purging.
					models_subset_masks = [self.purge_op_local(lmodel, gmodel, round_id) for lmodel in models_subset]
					# make sure that the local model values get to be zero after purging
					for masks, model in zip(models_subset_masks, models_subset):
						PurgeOps.apply_model_masks(model, masks)

					if self.fine_tuning_epochs > 0:
						# We need to make sure we have the intersection of global and local masks!
						fine_tuning_masks_after_local_purging = []
						for model_masks in models_subset_masks:
							new_model_masks = []
							for m_idx, (local_mask, global_mask) in enumerate(model_masks, gmodel_binary_masks):
								new_model_masks.append(local_mask | global_mask)
							fine_tuning_masks_after_local_purging.append(new_model_masks)

						# fine-tuning on purged models
						models_subset, models_subset_pseudogradients = \
							self.train_models(models_subset,
											  self.fine_tuning_epochs,
											  x_train_chunks_subset,
											  y_train_chunks_subset,
											  global_model=gmodel,
											  models_masks=fine_tuning_masks_after_local_purging)
				else:
					models_subset_masks = [ModelState.get_model_binary_masks(lmodel) for lmodel in models_subset]

				# save local models masks
				# for lidx, lmodel in enumerate(models_subset):
				# 	lmodel_binary_masks = ModelState.get_model_binary_masks(lmodel)
				# 	np.savez(self.local_model_fname_template.format(lidx, round_id), *lmodel.get_weights())
				# 	np.savez(self.local_model_masks_fname_template.format(lidx, round_id), *lmodel_binary_masks)

				# collection with local model sizes after purging for the current round
				non_zero_elements_after_purging = [ModelState.count_non_zero_elems(model) for model in models_subset]
				CustomLogger.info("Local models NNZ-params after purging: {}".format(non_zero_elements_after_purging))

				# evaluation of local models on local training and global test set
				local_models_train_loss, local_models_train_score, local_models_test_loss, local_models_test_score = \
					self.evaluate_models(models_subset, x_train_chunks, y_train_chunks, x_test, y_test,
										 models_subset_masks)

				self.merge_op.set_scaling_factors(scaling_factors_subset)
				global_weights = self.merge_op(models_subset, models_subset_masks,
											   global_model=gmodel,
											   models_pseudogradients=models_subset_pseudogradients)
				# set global model weights
				gmodel.set_weights(global_weights)

				if self.purge_op_global is not None and round_id > self.start_purging_at_round:
					# purge function __call__() definition: (purging_model, global_model, federation_round)
					# PurgeOps.apply_model_masks(gmodel, gmodel_binary_masks)
					gmodel_binary_masks = self.purge_op_global(gmodel, gmodel, round_id)
					PurgeOps.apply_model_masks(gmodel, gmodel_binary_masks)

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
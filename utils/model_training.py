import json
import random
import time

import numpy as np
import tensorflow as tf

from collections import namedtuple
from simulatedFL.models.model import Model
from simulatedFL.utils.model_merge import MergeOps
from simulatedFL.utils.model_purge import PurgeOps
from simulatedFL.utils.masked_callback import MaskedCallback

FederationRoundMetadata = namedtuple(typename='FederationRoundMetadata', field_names=['round_id', 'processing_time',
																					  'loss', 'score',
																					  'global_model_total_params',
																					  'local_models_total_params'])

class ModelTraining:

	class CentralizedTraining:

		def __init__(self, epochs=200, batch_size=100):
			self.epochs = epochs
			self.batch_size = batch_size

		def start(self, get_model_fn, x_train, y_train, x_test, y_test, info="Some Feedback"):
			"""Train."""
			print(f'    {info}')
			model = get_model_fn()
			epochs_test_eval_loss, epochs_test_eval_score = [], []
			for i in range(self.epochs):
				train_results = model.fit(x_train, y_train, epochs=1, batch_size=self.batch_size, verbose=True)
				eval_loss, eval_score = model.evaluate(x_test, y_test, batch_size=self.batch_size)
				epochs_test_eval_loss.append(eval_loss)
				epochs_test_eval_score.append(eval_score)
			eval_results = {"loss": epochs_test_eval_loss, "score":epochs_test_eval_score}
			return eval_results

	# class FederationRoundMetadata:
	# 	def __init__(self, round_id, processing_time, loss, score, global_model_total_params,
	# 				 local_models_total_params):
	# 		self.round_id = round_id
	# 		self.processing_time = processing_time
	# 		self.loss = loss
	# 		self.score = score
	# 		self.global_model_total_params = global_model_total_params
	# 		self.local_models_total_params = local_models_total_params
	#
	# 	def to_json(self):
	# 		return {"round_id": self.round_id, "processing_time": self.processing_time, "loss": self.loss,
	# 				"score": self.score, "global_model_total_params": self.global_model_total_params,
	# 				"local_models_total_params": self.local_models_total_params}

	class FederatedTraining:

		def __init__(self, merge_op, learners_num, rounds_num, local_epochs, learners_scaling_factors,
					 participation_rate=1, batch_size=100, purge_op=None, fine_tuning_epochs=0,
					 initialization_state=Model.InitializationStates.RANDOM,
					 burnin_period_epochs=0, round_robin_period_epochs=0):

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
			self.purge_op = purge_op
			self.fine_tuning_epochs = fine_tuning_epochs

			# Optional attributes related to federated model initialization state experiments.
			self.initialization_state = initialization_state
			self.burnin_period_epochs = burnin_period_epochs
			self.round_robin_period_epochs = round_robin_period_epochs

			# A json representation of the federated session with its associated training results.
			self.execution_stats = self.construct_federated_execution_stats()

		def construct_federated_execution_stats(self):
			res = {
				"federated_environment" : {
					"merge_function": None if self.merge_op is None else str(self.merge_op.__class__.__name__),
					"number_of_learners" : self.learners_num,
					"federation_rounds": self.rounds_num,
					"local_epochs_per_client": self.local_epochs,
					"scaling_factors": self.learners_scaling_factors,
					"participation_rate": self.participation_rate,
					"batch_size_per_client" : self.batch_size,
					"purge_function": None if self.purge_op is None else str(self.purge_op.__class__.__name__),
					"fine_tuning_epochs": self.fine_tuning_epochs,
					"federated_model_initialization_state" : self.initialization_state,
					"burnin_period_epochs" : self.burnin_period_epochs,
					"round_robin_period_epochs": self.round_robin_period_epochs
				},
				"federated_execution_results" : list()
			}
			return res


		def set_initial_federation_model_state(self, gmodel, lmodels, x_train_chunks, y_train_chunks):

			model_state = []
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


		def count_non_zero_elems(self, model):
			return sum([np.count_nonzero(matrix) for matrix in model.get_weights()])


		def train_local_models(self, lmodels, epochs_num, x_train_chunks, y_train_chunks, models_masks=None):
			for lidx, lmodel in enumerate(lmodels):

				callbacks = []
				if models_masks is not None:
					model_mask = models_masks[lidx]
					callbacks.append(MaskedCallback(model_masks=model_mask))

				lmodel.fit(
					x_train_chunks[lidx], y_train_chunks[lidx],
					epochs=epochs_num, batch_size=self.batch_size, verbose=False, callbacks=callbacks
				)
			return lmodels


		def start(self, get_model_fn, x_train_chunks, y_train_chunks, x_test, y_test, info="Some Feedback"):
			"""Federated Training."""

			lmodels = [get_model_fn() for _ in range(self.learners_num)]
			gmodel = get_model_fn()
			print(f'    {info}')

			federation_rounds_stats = []
			st = time.process_time()
			global_weights = self.set_initial_federation_model_state(gmodel, lmodels, x_train_chunks, y_train_chunks)
			global_model_num_params = 0
			local_models_num_params = []

			for round_id in range(self.rounds_num + 1):

				# set global model weights
				gmodel.set_weights(global_weights)
				global_model_num_params = self.count_non_zero_elems(gmodel)

				# eval global model
				loss, score = gmodel.evaluate(x_test, y_test, verbose=False)
				# capture end time
				et = time.process_time()

				# raw elapsed time
				elapsed_time = et - st
				# avg elapsed time -- assuming every learner was running in parallel
				avg_elapsed_time = np.divide(elapsed_time, len(lmodels))
				federation_round_metadata = FederationRoundMetadata(round_id=round_id, processing_time=avg_elapsed_time,
																	loss=loss,
																	score=score,
																	global_model_total_params=global_model_num_params,
																	local_models_total_params=local_models_num_params)
				federation_rounds_stats.append(federation_round_metadata._asdict())
				print("Federation Round: {}, Loss: {}, Score: {}".format(round_id, loss, score))

				# initialize execution time
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

				# normal training
				models_subset = self.train_local_models(models_subset, self.local_epochs,
														x_train_chunks_subset, y_train_chunks_subset)

				# compute training masks after purging operation
				models_subset_masks = [self.purge_op(model, round_id) for model in models_subset]

				# apply purging masks and fine-tune
				models_subset = self.train_local_models(models_subset, self.fine_tuning_epochs,
														x_train_chunks_subset, y_train_chunks_subset,
														models_masks=models_subset_masks)

				# collection with local model sizes for the current round
				local_models_num_params = []
				for model in models_subset:
					local_models_num_params.append(self.count_non_zero_elems(model))

				print("Total Local Models: ", len(models_subset))
				self.merge_op.set_scaling_factors(scaling_factors_subset)
				global_weights = self.merge_op(models_subset)

			# clear memory
			del lmodels
			del gmodel
			tf.keras.backend.clear_session()

			self.execution_stats["federated_execution_results"] = federation_rounds_stats
			return self.execution_stats
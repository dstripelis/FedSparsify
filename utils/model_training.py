import random
import time

import numpy as np
import tensorflow as tf

from simulatedFL.models.model import Model
from simulatedFL.utils.model_merge import MergeOps


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


	class FederatedTraining:

		def __init__(self, learners_num=100, rounds_num=100, participation_rate=1, local_epochs=4, batch_size=100,
					 learners_scaling_factors=list(), initialization_state=Model.InitializationStates.RANDOM,
					 burnin_period_epochs=0, round_robin_period_epochs=0):
			self.learners_num = learners_num
			self.rounds_num = rounds_num
			self.participation_rate = participation_rate
			self.local_epochs = local_epochs
			self.batch_size = batch_size

			if len(learners_scaling_factors) == 0:
				self.learners_scaling_factors = [1] * learners_num
			else:
				self.learners_scaling_factors = learners_scaling_factors

			self.initialization_state = initialization_state
			self.burnin_period_epochs = burnin_period_epochs
			self.round_robin_period_epochs = round_robin_period_epochs
			self.execution_stats = self.construct_federated_execution_stats()


		def construct_federated_execution_stats(self):
			res = {
				"federated_environment" : {
					"number_of_learners" : self.learners_num,
					"federation_rounds": self.rounds_num,
					"participation_rate": self.participation_rate,
					"local_epochs_per_client" : self.local_epochs,
					"batch_size_per_client" : self.batch_size,
					"federated_model_initialization_state" : self.initialization_state,
					"burnin_period_epochs" : self.burnin_period_epochs,
					"round_robin_period_epochs": self.round_robin_period_epochs
				},
				"federated_execution_results" : list()
			}
			return res


		def set_initial_model_state(self, gmodel, lmodels, x_train_chunks, y_train_chunks):

			model_state = []
			if self.initialization_state == Model.InitializationStates.RANDOM:
				# Random Initialization - Assign the random state to every learner - see final loop at the end.
				model_state = gmodel.get_weights()

			elif self.initialization_state == Model.InitializationStates.BURNIN_SINGLETON:
				# Burnin Initialization - 'Burn' a single learner and assign its weight - see final loop at the end.
				lmodels[0].fit(x_train_chunks[0], y_train_chunks[0],
					epochs=self.burnin_period_epochs, batch_size=self.batch_size, verbose=False
				)
				model_state = lmodels[0].get_weights()

			elif self.initialization_state == Model.InitializationStates.BURNIN_MEAN_CONSENSUS:
				# Burnin Initialization - 'Burn' all learners - average them and assign their weight - see final loop at the end.
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
				# Burnin Initialization - 'Burn' all learners and scale them - see final loop at the end.
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

			elif self.initialization_state == Model.InitializationStates.ROUND_ROBIN:
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


		def train_local_models(self, lmodels, x_train_chunks, y_train_chunks):
			for lidx, lmodel in enumerate(lmodels):
				lmodel.fit(
					x_train_chunks[lidx], y_train_chunks[lidx],
					epochs=self.local_epochs, batch_size=self.batch_size, verbose=False
				)
			return lmodels


		def start(self, get_model_fn, x_train_chunks, y_train_chunks, x_test, y_test, info="Some Feedback"):
			"""Federated Training."""

			lmodels = [get_model_fn() for _ in range(self.learners_num)]
			gmodel = get_model_fn()
			print(f'    {info}')

			federation_rounds_stats = []
			st = time.process_time()
			global_weights = self.set_initial_model_state(gmodel, lmodels, x_train_chunks, y_train_chunks)

			for r in range(self.rounds_num + 1):

				# set global model weights
				gmodel.set_weights(global_weights)

				# eval global model
				loss, score = gmodel.evaluate(x_test, y_test, verbose=False)
				# capture end time
				et = time.process_time()

				# raw elapsed time
				elapsed_time = et - st
				# avg elapsed time -- assuming every learner was running in parallel
				avg_elapsed_time = np.divide(elapsed_time, len(lmodels))
				federation_rounds_stats.append((avg_elapsed_time, loss, score))
				print("Federation Round: {}, Loss: {}, Score: {}".format(r, loss, score))

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
				models_subset = self.train_local_models(models_subset, x_train_chunks_subset, y_train_chunks_subset)
				print("Total Local Models: ", len(models_subset))
				global_weights = MergeOps.merge_local_models(models_subset, scaling_factors_subset)

			# Clear memory
			del lmodels
			del gmodel
			tf.keras.backend.clear_session()

			self.execution_stats["federated_execution_results"] = federation_rounds_stats
			return self.execution_stats
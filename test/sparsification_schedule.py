import matplotlib

import matplotlib.pyplot as plt
import numpy as np


def gradual_exponential_pruning(exp=3):
	"""
	Gradual Pruning Equation:
		s_t = s_f + (s_i - s_f)(1 - ((t-t0)/(T-t0))^exp, with t, T >= t0

	Notation:
		s_t: sparsity level at current round/iteration/epoch
		t: current round
		s_i: initial sparsity level
		s_f: final sparsity level
		t0: which round to start sparsification/pruning
		T: total number of rounds/iterations/epochs
	"""
	def sparsity_level_fn(t, si, sf, t0, T, f, exp):
		# st = sf + (si - sf) * np.power(1 - np.divide(f * (np.floor(t/f)) - t0, T - t0), exp)
		if t % f == 0:
			st = sf + (si - sf) * np.power(1 - np.divide(t - t0, T-t0), exp)
		else:
			t = (t // f) * f
			st = sf + (si - sf) * np.power(1 - np.divide(t - t0, T - t0), exp)
		return st

	# Variables:
	# 	initial_sparsity, final_sparsity, start_pruning_at_round, frequency
	combinations = [
		(0, 95, 1, 1),
		(0, 95, 1, 2),
		(0, 95, 1, 5),
		(0, 95, 1, 10),
		(0, 95, 1, 15),
		(0, 95, 1, 20),
	]


	schedules = dict()
	T = 200
	for comb in combinations:
		initial_sparsity, final_sparsity, start_pruning_at_round, frequency = comb[0], comb[1], comb[2], comb[3]
		sparsity_levels = []
		for t in range(T):
			if t >= start_pruning_at_round:
				sparsity_level = sparsity_level_fn(
					t, initial_sparsity, final_sparsity, start_pruning_at_round, T, frequency, exp=exp)
				sparsity_levels.append(sparsity_level)
			else:
				sparsity_levels.append(0)
		print(comb, sparsity_levels)
		schedules[comb] = sparsity_levels

	for schedule_key, schedule_val in schedules.items():
		plt.plot(range(T), schedule_val, label=schedule_key)
	# plt.title(r"Sparsify every $dt$:{} round".format(1))
	plt.ylabel("Sparsity Level")
	plt.xlabel("Federation Round")
	plt.title("(InitialSparsity, FinalSparsity, StartPruningAt, PruneEvery)")
	plt.legend()
	plt.savefig("Gradual Pruning Schedule")


def gradual_step_wise_pruning():
	sparsity_level_fn = lambda t, si, sf, f, T: \
		si + ((sf - si) / np.ceil(T / f)) * np.ceil(t/f)

	# variables:
	# 	initial_sparsity, final_sparsity, prune_every
	combinations = [
		(0, 95, 1),
		(0, 95, 2),
		(0, 95, 5),
		(0, 95, 10),
		(0, 95, 15),
		(0, 95, 20),
		(0, 95, 50),
	]

	schedules = dict()
	T = 200
	for comb in combinations:
		sparsity_levels = []
		for t in range(T):
			sparsity_level = sparsity_level_fn(t, comb[0], comb[1], comb[2], T)
			sparsity_levels.append(sparsity_level)
		print(comb, sparsity_levels)
		schedules[comb] = sparsity_levels

	for schedule_key, schedule_val in schedules.items():
		plt.plot(range(T), schedule_val, label=schedule_key)
	plt.ylabel("Sparsity Level")
	plt.xlabel("Federation Round")
	plt.title("(InitialSparsity, FinalSparsity, PruneEveryRounds)")
	plt.legend()
	plt.savefig("Step-Wise Schedule")
	plt.show()


def gradual_nnz_pruning():

	# def prune_at_round(nnz, T, s=0.02):
	# 	summation = 0
	# 	for k in range(1, T):
	# 		summation += np.ceil(np.power(-1, k-1) * nnz * np.power(s, k))
	# 	return summation

	def nnz_at_round(nnz_init, T, s=0.02):
		nnz_rounds = {0: nnz_init}
		for i in range(1, T+1):
			nnz_rounds[i] = nnz_rounds[i-1] - np.floor(nnz_rounds[i-1] * s)
		return nnz_rounds[T]



if __name__ == "__main__":
	gradual_exponential_pruning(exp=12)
	# gradual_exponential_pruning(exp=6)
	gradual_exponential_pruning(exp=3)
	gradual_exponential_pruning(exp=1)
	# gradual_exponential_pruning(exp=10)
	# gradual_step_wise_pruning()
	# gradual_nnz_pruning()
	# fn = lambda s: 1 - 5*s + 10*np.power(s, 2) -10*np.power(s, 3) + 5*np.power(s, 4) - np.power(s, 5)
	# print(fn(0.02))
	# print(fn(0.04))
	# print(fn(0.08))
	# print(fn(0.09))
	# print(fn(0.1))
	# print(fn(0.5))
	plt.show()
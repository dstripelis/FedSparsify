import glob
import json

import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages


def plot_3x3(learners_execution_stats, plot_by_round=False, plot_by_epoch=False):
	fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(12, 12))
	plt.subplots_adjust(wspace=0.15, hspace=0.3)

	sorted_learner_num = sorted(learners_execution_stats.keys())

	for lidx, learner_num in enumerate(sorted_learner_num):

		axes[lidx, 1].set_title("Total Learners: {}".format(learner_num))

		stats = learners_execution_stats[learner_num]
		sorted_prate_num = sorted(stats.keys(), reverse=True)
		for pidx, prate_num in enumerate(sorted_prate_num):

			if lidx == 0:
				if pidx == 1:
					axes[lidx, pidx].set_title(
						"Total Learners: {}\nParticipation Rate: {}".format(learner_num, prate_num))
				else:
					axes[lidx, pidx].set_title("Participation Rate: {}".format(prate_num))

			axes[lidx, pidx].set_ylim(0.0, 0.95)
			linestyle = "solid"

			random_env = stats[prate_num]['random']['federated_environment']
			random_res = stats[prate_num]['random']['federated_execution_results'][:501]
			if plot_by_round:
				random_x_axis = [i for i in range(len(random_res))]
			if plot_by_epoch:
				random_x_axis = [(i + 1) * random_env['local_epochs_per_client'] for i in range(len(random_res))]
			random_y_axis = [x[2] for x in random_res]
			random_line, = axes[lidx, pidx].plot(random_x_axis, random_y_axis, color="red", linestyle=linestyle)

			burnin_consensus_mean_env = stats[prate_num]['burnin_mean_consensus']['federated_environment']
			burnin_consensus_mean_res = stats[prate_num]['burnin_mean_consensus']['federated_execution_results'][:501]
			if plot_by_round:
				burnin_consensus_mean_x_axis = [i for i in range(len(burnin_consensus_mean_res))]
			if plot_by_epoch:
				burnin_period = burnin_consensus_mean_env['burnin_period_epochs']
				burnin_consensus_mean_x_axis = [burnin_period]
				burnin_consensus_mean_x_axis.extend(
					[burnin_period + (i + 1) * random_env['local_epochs_per_client'] for i in
					 range(len(burnin_consensus_mean_res) - 1)])
			burnin_consensus_mean_y_axis = [x[2] for x in burnin_consensus_mean_res]
			burnin_consensus_line, = axes[lidx, pidx].plot(burnin_consensus_mean_x_axis, burnin_consensus_mean_y_axis,
														   color="blue",
														   linestyle=linestyle)

			burnin_singleton_env = stats[prate_num]['burnin_singleton']['federated_environment']
			burnin_singleton_res = stats[prate_num]['burnin_singleton']['federated_execution_results'][:501]
			if plot_by_round:
				burnin_singleton_x_axis = [i for i in range(len(burnin_singleton_res))]
			if plot_by_epoch:
				burnin_period = burnin_singleton_env['burnin_period_epochs']
				burnin_singleton_x_axis = [burnin_period]
				burnin_singleton_x_axis.extend([burnin_period + (i + 1) * random_env['local_epochs_per_client'] for i in
												range(len(burnin_singleton_res) - 1)])
			burnin_singleton_y_axis = [x[2] for x in burnin_singleton_res]
			burnin_singleton_line, = axes[lidx, pidx].plot(burnin_singleton_x_axis, burnin_singleton_y_axis,
														   color="green",
														   linestyle=linestyle)

			if 'round_robin_all_learners' in stats[prate_num]:
				round_robin_all_learners_env = stats[prate_num]['round_robin_all_learners']['federated_environment']
				round_robin_all_learners_res = stats[prate_num]['round_robin_all_learners'] \
												   ['federated_execution_results'][:501]
				if plot_by_round:
					round_robin_all_learners_x_axis = [i for i in range(len(round_robin_all_learners_res))]
				if plot_by_epoch:
					round_robin_epoch_period = round_robin_all_learners_env['round_robin_period_epochs'] * \
											   round_robin_all_learners_env['number_of_learners']
					round_robin_all_learners_x_axis = [
						round_robin_epoch_period + (i + 1) * random_env['local_epochs_per_client'] for i in
						range(len(round_robin_all_learners_res))]
				round_robin_all_learners_y_axis = [x[2] for x in round_robin_all_learners_res]
				round_robin_all_learners_line, = axes[lidx, pidx].plot(round_robin_all_learners_x_axis,
																	   round_robin_all_learners_y_axis,
																	   color="orange",
																	   linestyle=linestyle)

			if 'round_robin_rate_sample' in stats[prate_num]:
				round_robin_rate_sample_env = stats[prate_num]['round_robin_rate_sample']['federated_environment']
				round_robin_rate_sample_res = stats[prate_num]['round_robin_rate_sample'] \
												  ['federated_execution_results'][:501]
				if plot_by_round:
					round_robin_rate_sample_x_axis = [i for i in range(len(round_robin_rate_sample_res))]
				if plot_by_epoch:
					round_robin_epoch_period = round_robin_rate_sample_env['round_robin_period_epochs'] * \
											   (round_robin_rate_sample_env['participation_rate'] *
												round_robin_rate_sample_env['number_of_learners'])
					round_robin_rate_sample_x_axis = [
						round_robin_epoch_period + (i + 1) * round_robin_rate_sample_env['local_epochs_per_client']
						for i in range(len(round_robin_rate_sample_res))]
				round_robin_rate_sample_y_axis = [x[2] for x in round_robin_rate_sample_res]
				round_robin_rate_sample_line, = axes[lidx, pidx].plot(round_robin_rate_sample_x_axis,
																	  round_robin_rate_sample_y_axis,
																	  color="indigo",
																	  linestyle=linestyle)

			if lidx == 0 and pidx == 0:
				# legend_handlers = [random_line, burnin_consensus_line, burnin_singleton_line]
				# legend_handlers_labels = ["Random", "Burnin Mean Consensus", "Burnin Singleton"]
				legend_handlers = [random_line, burnin_consensus_line, burnin_singleton_line, round_robin_rate_sample_line]
				legend_handlers_labels = ["Random", "Burnin Mean Consensus", "Burnin Singleton", "Round-Robin (sampled)"]
				# legend_handlers = [random_line, burnin_consensus_line, burnin_singleton_line,
				# 				   round_robin_all_learners_line, round_robin_rate_sample_line]
				# legend_handlers_labels = ["Random", "Burnin Mean Consensus", "Burnin Singleton", "Round-Robin (all)",
				# 						  "Round-Robin (sampled)"]


	# Set common axis labels
	# fig.text(0.5, 0.11, 'Federation Rounds', ha='center', va='center', fontsize=12)
	# fig.text(0.5, 0.135, 'Federation Rounds', ha='center', va='center', fontsize=12)
	fig.text(0.5, 0.135, 'Local Epochs (Parallel)', ha='center', va='center', fontsize=12)
	fig.text(0.08, 0.5, 'Test Accuracy', ha='center', va='center', rotation='vertical', fontsize=12)
	# Set legend
	fig.subplots_adjust(bottom=0.18)
	fig.legend(handles=legend_handlers, labels=legend_handlers_labels, loc='lower center',
			   fancybox=False, shadow=False, fontsize=12, ncol=1)
	return fig


def plot_3x3_burnin_periods(learners_execution_stats):
	fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(12, 12))
	plt.subplots_adjust(wspace=0.15, hspace=0.3)

	legend_handlers = []
	legend_handlers_labels = []
	sorted_learner_num = sorted(learners_execution_stats.keys())

	for lidx, learner_num in enumerate(sorted_learner_num):

		axes[lidx, 1].set_title("Total Learners: {}".format(learner_num))

		stats = learners_execution_stats[learner_num]
		sorted_prate_num = sorted(stats.keys(), reverse=True)
		for pidx, prate_num in enumerate(sorted_prate_num):

			if lidx == 0:
				if pidx == 1:
					axes[lidx, pidx].set_title(
						"Total Learners: {}\nParticipation Rate: {}".format(learner_num, prate_num))
				else:
					axes[lidx, pidx].set_title("Participation Rate: {}".format(prate_num))

			axes[lidx, pidx].set_ylim(0.0, 0.95)

			burnins = learners_execution_stats[learner_num][prate_num]['burnin_singleton']
			# burnins = learners_execution_stats[learner_num][prate_num]['burnin_mean_consensus']
			sorted_burnins = sorted(burnins.keys(), reverse=True)

			for burnin in sorted_burnins:
				linestyle = "solid"
				print(burnin)
				burnin_singleton = stats[prate_num]['burnin_singleton'][burnin][:501]
				# burnin_singleton = stats[prate_num]['burnin_mean_consensus'][burnin][:501]

				burnin_singleton_line, = axes[lidx, pidx].plot([i for i in range(len(burnin_singleton))],
															   [x[2] for x in burnin_singleton],
															   linestyle=linestyle)

				if lidx == 0 and pidx == 0:
					legend_handler_label = "Burnin Singleton Period: {}".format(burnin)
					# legend_handler_label = "Burnin Mean Consensus Period: {}".format(burnin)
					legend_handlers.append(burnin_singleton_line)
					legend_handlers_labels.append(legend_handler_label)

	# Set common axis labels
	fig.text(0.5, 0.11, 'Federation Rounds', ha='center', va='center', fontsize=12)
	fig.text(0.08, 0.5, 'Test Accuracy', ha='center', va='center', rotation='vertical', fontsize=12)
	# Set legend
	fig.subplots_adjust(bottom=0.15)
	print(legend_handlers, legend_handlers_labels)
	fig.legend(handles=legend_handlers, labels=legend_handlers_labels, loc='lower center',
			   fancybox=False, shadow=False, fontsize=12, ncol=1)
	return fig


if __name__ == "__main__":

	rounds_num = 200
	learners_num_list = [10, 100, 1000]

	# random_init_dir = "FashionMNIST/non-iid/classes2/rounds500"
	# burnin_init_dir = "FashionMNIST/non-iid/classes2/rounds500"
	# random_init_dir = "FashionMNIST/iid/rounds1000"
	# burnin_init_dir = "FashionMNIST/iid/rounds500"

	random_init_dir = "Cifar10/iid/rounds500"
	burnin_init_dir = "Cifar10/iid/rounds500"

	# random_init_dir = "IMDB/iid/rounds500"
	# burnin_init_dir = "IMDB/iid/rounds500"

	files = []
	files = glob.glob("../logs/FashionMNIST/*.json")
	print(len(files))
	# files = glob.glob("../logs/{}/*init_random*.json".format(random_init_dir))
	# files.extend(glob.glob("../logs/{}/*singleton*burnin_5.json".format(burnin_init_dir)))
	# files.extend(glob.glob("../logs/FashionMNIST/iid/rounds1000/*singleton*burnin_5.json".format(burnin_init_dir)))
	# files.extend(glob.glob("../logs/{}/*singleton*burnin_10.json".format(burnin_init_dir)))
	# files.extend(glob.glob("../logs/{}/*singleton*burnin_25.json".format(burnin_init_dir)))
	# files.extend(glob.glob("../logs/{}/*singleton*burnin_50.json".format(burnin_init_dir)))
	# files.extend(glob.glob("../logs/{}/*init_burnin_mean_consensus*.json".format(burnin_init_dir)))
	# files.extend(glob.glob("../logs/{}/*init_burnin_singleton*.json".format(burnin_init_dir)))
	# files.extend(glob.glob("../logs/{}/*init_round_robin*.json".format(burnin_init_dir)))

	learners_stats = dict()
	learners_stats_burnin = dict()
	for learners_num in learners_num_list:
		learners_stats[learners_num] = dict()
		learners_stats_burnin[learners_num] = dict()
		for f in files:
			if '.learners_{}.'.format(learners_num) in f:
				json_content = json.load(open(f))
				prate = json_content["federated_environment"]["participation_rate"]
				init_state = json_content["federated_environment"]["federated_model_initialization_state"]
				burnin_period_epochs = int(json_content["federated_environment"]["burnin_period_epochs"])

				""" learners_stats collection """
				if prate not in learners_stats[learners_num]:
					learners_stats[learners_num][prate] = dict()
				learners_stats[learners_num][prate][init_state] = json_content

				""" learners_stats_burnin collection """
				if prate not in learners_stats_burnin[learners_num]:
					learners_stats_burnin[learners_num][prate] = dict()

				if init_state not in learners_stats_burnin[learners_num][prate]:
					learners_stats_burnin[learners_num][prate][init_state] = dict()

				if burnin_period_epochs not in learners_stats_burnin[learners_num][prate][init_state]:
					learners_stats_burnin[learners_num][prate][init_state][burnin_period_epochs] = dict()

				learners_stats_burnin[learners_num][prate][init_state][burnin_period_epochs] = json_content

				print(f)
				print(learners_num, prate, init_state)
				print(learners_stats[learners_num][prate][init_state])

	fig = plot_3x3(learners_stats, plot_by_epoch=True)
	# fig = plot_3x3_burnin_periods(learners_stats_burnin)
	file_out = "{}.pdf".format("InitializationStatesConvergence")
	pdfpages1 = PdfPages(file_out)
	pdfpages1.savefig(figure=fig, bbox_inches='tight')
	pdfpages1.close()
	plt.show()

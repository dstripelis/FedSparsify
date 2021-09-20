import glob
import json

import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages


def plot_3x3(learners_execution_stats):

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
			if prate_num == 1:
				linestyle = "solid"
			# elif prate_num == 0.5:
			# 	linestyle = "dashed"
			# elif prate_num == 0.1:
			# 	linestyle = "dotted"

			random = stats[prate_num]['random']
			burnin_consensus_mean = stats[prate_num]['burnin_mean_consensus']
			burnin_singleton = stats[prate_num]['burnin_singleton']

			random_line, = axes[lidx, pidx].plot([i for i in range(len(random))],
												 [x[2] for x in random],
												 color="red",
												 linestyle=linestyle)
			burin_consensus_line, = axes[lidx, pidx].plot([i for i in range(len(burnin_consensus_mean))],
														 [x[2] for x in burnin_consensus_mean],
														 color="blue",
														 linestyle=linestyle)
			burnin_singleton_line, = axes[lidx, pidx].plot([i for i in range(len(burnin_singleton))],
														  [x[2] for x in burnin_singleton],
														  color="green",
														  linestyle=linestyle)

			if lidx == 0 and pidx == 0:
				legend_handlers = [random_line, burin_consensus_line, burnin_singleton_line]
				legend_handlers_labels = ["Random", "Burnin Mean Consensus", "Burnin Singleton"]

	# Set common axis labels
	fig.text(0.5, 0.11, 'Federation Rounds', ha='center', va='center', fontsize=12)
	fig.text(0.08, 0.5, 'Test Accuracy', ha='center', va='center', rotation='vertical', fontsize=12)
	# Set legend
	fig.subplots_adjust(bottom=0.15)
	fig.legend(handles=legend_handlers, labels=legend_handlers_labels, loc='lower center',
			   fancybox=False, shadow=False, fontsize=12, ncol=1)
	return fig


if __name__ == "__main__":

	rounds_num = 200
	learners_num_list = [10, 100, 1000]

	files = glob.glob("../logs/FashionMNIST/*.json")

	learners_stats = dict()
	for learners_num in learners_num_list:
		learners_stats[learners_num] = dict()
		for f in files:
			if '.learners_{}.'.format(learners_num) in f:
				json_content = json.load(open(f))
				prate = json_content["federated_environment"]["participation_rate"]
				init_state = json_content["federated_environment"]["federated_model_initialization_state"]
				if prate not in learners_stats[learners_num]:
					learners_stats[learners_num][prate] = dict()
				learners_stats[learners_num][prate][init_state] = json_content['federated_execution_results']
				print(f)
				print(learners_num, prate, init_state)
				print(learners_stats[learners_num][prate][init_state])

	fig = plot_3x3(learners_stats)
	file_out = "{}.pdf".format("InitializationStatesConvergence")
	pdfpages1 = PdfPages(file_out)
	pdfpages1.savefig(figure=fig, bbox_inches='tight')
	pdfpages1.close()
	plt.show()

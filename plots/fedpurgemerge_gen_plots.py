import glob
import json

import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from itertools import groupby


def plot_communication_cost_results(files, federation_round=False, transmission_cost=False):

	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
	bits_per_param = 32
	line_styles = ["solid", "--", "-."]

	for fidx, f in enumerate(files):
		json_content = json.load(open(f))
		number_of_learners = json_content["federated_environment"]["number_of_learners"]
		local_epochs = int(json_content["federated_environment"]["local_epochs_per_client"])
		prate = json_content["federated_environment"]["participation_rate"]
		sparsity = float(json_content["federated_environment"]["sparsity_level"])
		fine_tuning_epochs = float(json_content["federated_environment"]["fine_tuning_epochs"])
		federated_execution_results = json_content["federated_execution_results"]
		tupled_res = []
		incremental_comm_cost = 0
		for res in federated_execution_results:
			total_participating_learners = len(res['local_models_total_params'])
			# communication cost per round in terms of model exchanges is: total_local_models + number_of_learners * global model
			comm_cost = sum(res['local_models_total_params']) + res['global_model_total_params'] * total_participating_learners
			comm_cost *= bits_per_param
			comm_cost /= 8e+6
			incremental_comm_cost += comm_cost
			tupled_res.append((res['global_model_test_score'], local_epochs, sparsity, fine_tuning_epochs, incremental_comm_cost))

		y_axis = [res[0] for res in tupled_res]
		if transmission_cost:
			ax.set_xlabel('Transmission Cost (#MBs)')
			x_axis = [res[4] for res in tupled_res]
		if federation_round:
			ax.set_xlabel('Federation Round')
			x_axis = [x for x in range(len(y_axis))]

		line_label = "le: {}, s:{}, ft: {}".format(local_epochs, sparsity, fine_tuning_epochs)
		print(line_label)
		ax.plot(x_axis, y_axis, label=line_label)

	# ax.set_xticks(sorted(set(x_ticks)))
	ax.set_ylabel('Top-1 Accuracy')
	fig.legend(title='Training Specs')

	return fig


def plot_sparsified_results(files):

	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))

	results = []
	for f in files:
		json_content = json.load(open(f))
		number_of_learners = json_content["federated_environment"]["number_of_learners"]
		local_epochs = int(json_content["federated_environment"]["local_epochs_per_client"])
		prate = json_content["federated_environment"]["participation_rate"]
		sparsity = float(json_content["federated_environment"]["sparsity_level"])
		fine_tuning_epochs = float(json_content["federated_environment"]["fine_tuning_epochs"])
		federated_execution_results = json_content["federated_execution_results"]
		tupled_res = [(res['global_model_test_score'], local_epochs, sparsity, fine_tuning_epochs, res['global_model_total_params'])
					  for res in federated_execution_results[1:]]
		max_res = max(tupled_res, key=lambda x: x[0])
		results.append(max_res)

	results.sort(key=lambda x: (x[1], x[3]))
	fine_tuning_groups = groupby(results, key=lambda x: x[3])
	x_ticks = []
	colors = ["red", "blue", "green", "orange"]
	markers = ["x", "o", "d", "p"]
	for group_id, (fine_tuning_epochs, group) in enumerate(fine_tuning_groups):
		y_axis = []
		x_axis = []
		num_params = []

		local_epochs = 0
		for res in group:
			y_axis.append(res[0])
			x_axis.append(res[2])
			x_ticks.append(res[2])
			num_params.append(res[4])
			local_epochs = res[1]

		line_label = "le: {}, ft: {}".format(local_epochs, fine_tuning_epochs)
		print(group_id, line_label)
		for idx, (x, y, p) in enumerate(zip(x_axis, y_axis, num_params)):
			if idx == 0:
				ax.scatter(x, y, color=colors[group_id], marker=markers[group_id], label=line_label)
			else:
				ax.scatter(x, y, color=colors[group_id], marker=markers[group_id])
			print(x,y)
			# for bold text use: weight='heavy'
			ax.annotate(str(f'{p:,}'), xy=(x + 0.005, y + 0.0005), fontsize="xx-small",
						verticalalignment="bottom",
						horizontalalignment='left')

	ax.set_xticks(sorted(set(x_ticks)))
	ax.set_ylabel('Top-1 Accuracy')
	ax.set_xlabel('Sparsity Level')
	fig.legend(title='Training Specs')

	return fig



if __name__ == "__main__":

	# files = glob.glob("../logs/FashionMNIST/centralized/purge_weight_magnitude/*.json")
	files = glob.glob("../logs/Cifar100/centralized/purge_weight_magnitude/*.json")
	# files = glob.glob("../logs/IMDB/centralized/purge_weight_magnitude/*.json")
	# files = glob.glob("../logs/FashionMNIST/non-iid/classes2/fedpurgemerge/purge_weight_magnitude_local_merge_weighted_avg/*.json")
	fig = plot_sparsified_results(files)
	# fig = plot_communication_cost_results(files, federation_round=False, transmission_cost=True)
	file_out = "{}.pdf".format("PurgeMerge")
	pdfpages1 = PdfPages(file_out)
	pdfpages1.savefig(figure=fig, bbox_inches='tight')
	pdfpages1.close()
	plt.show()

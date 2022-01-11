import glob
import json

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from itertools import groupby


def plot_centralized_epoch_results(files, x_axis_epoch=False, y_axis_secondary_params=False):

	fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
	ax1.set_ylim(0.0, 0.9)
	if y_axis_secondary_params:
		ax2 = ax1.twinx()
		ax2.set_ylabel('Global Model Params')
		# ax2.set_ylim(0, 120000)
		# ax2.set_ylim(0, 1e6)
		ax2.set_ylim(0, 4e6)
	bits_per_param = 32
	max_x = -1
	line_styles = ["solid", "--", "-."]

	for fidx, f in enumerate(files):
		json_content = json.load(open(f))
		total_epochs = int(json_content["federated_environment"]["federation_rounds"])
		batch_size = int(json_content["federated_environment"]["batch_size_per_client"])
		purging_function = json_content["federated_environment"]["purge_function_global"]
		start_pruning_at_epoch = int(json_content["federated_environment"]["start_purging_at_round"])
		sparsity_level = float(json_content["federated_environment"]["sparsity_level"])
		local_epochs = int(json_content["federated_environment"]["local_epochs_per_client"])
		fine_tuning_epochs = float(json_content["federated_environment"]["fine_tuning_epochs"])
		purge_function = str(json_content["federated_environment"]["purge_function_global"])
		num_epochs_before_pruning = 1 if (start_pruning_at_epoch == 0 and purging_function is not None) \
			else start_pruning_at_epoch
		num_epochs_after_pruning = total_epochs - num_epochs_before_pruning
		federated_execution_results = json_content["federated_execution_results"]
		tupled_res = []
		for res in federated_execution_results:
			global_model_params = res['global_model_total_params']
			tupled_res.append((res['global_model_test_score'], local_epochs, sparsity_level, fine_tuning_epochs,
							   global_model_params, num_epochs_after_pruning))

		y1_axis = [res[0] for res in tupled_res]
		if x_axis_epoch:
			ax1.set_xlabel('Epoch')
			x_axis = [x for x in range(len(y1_axis))]

		y1_axis = y1_axis[:200]
		x_axis = x_axis[:200]
		max_x = max(max(x_axis), max_x)

		line_label = "before-pruning: {}, after-pruning: {}, sparsification: {}"\
			.format(num_epochs_before_pruning, num_epochs_after_pruning, sparsity_level)
		# print(line_label, " max accuracy: {}".format(max(y1_axis)), files[fidx])
		# print([(idx, y) for idx, y in enumerate(y1_axis)])
		print(f)
		print(line_label)
		ax1.plot(x_axis, y1_axis, label=line_label)

		if y_axis_secondary_params:
			y2_axis = [res[4] for res in tupled_res]
			y2_axis = y2_axis[:200]
			ax2.plot(x_axis, y2_axis, linestyle="--")

	ax1.set_ylabel('Top-1 Accuracy')
	fig.subplots_adjust(bottom=0.2)
	# fig.legend(title='Training Specs')
	fig.legend(title='Training Epochs', loc="lower center", bbox_to_anchor=(0.44, 0.01), ncol=2)

	return fig


def plot_centralized_sparsified_results(files, progressive=True):

	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))

	results = []
	for f in files:
		json_content = json.load(open(f))
		total_epochs = int(json_content["federated_environment"]["federation_rounds"])
		batch_size = int(json_content["federated_environment"]["batch_size_per_client"])
		purging_function = json_content["federated_environment"]["purge_function_global"]
		start_pruning_at_epoch = int(json_content["federated_environment"]["start_purging_at_round"])
		sparsity_level = float(json_content["federated_environment"]["sparsity_level"])
		local_epochs = int(json_content["federated_environment"]["local_epochs_per_client"])
		fine_tuning_epochs = float(json_content["federated_environment"]["fine_tuning_epochs"])
		purge_function = str(json_content["federated_environment"]["purge_function_global"])
		original_model_size = float(json_content["federated_execution_results"][0]["global_model_total_params"])
		num_epochs_before_pruning = 1 if (start_pruning_at_epoch == 0 and purging_function is not None) \
			else start_pruning_at_epoch
		num_epochs_after_pruning = total_epochs - num_epochs_before_pruning
		federated_execution_results = json_content["federated_execution_results"]
		tupled_res = [(
			res['global_model_test_score'],
			int(num_epochs_before_pruning),
			sparsity_level,
			int(num_epochs_after_pruning),
			res['global_model_total_params'],
			purge_function,
			original_model_size,
			','.join(f.split("/")[-3:])
		) for res in federated_execution_results[num_epochs_before_pruning+1:]]
		if progressive:
			# if progressive we plot the final value
			max_res = tupled_res[-1]
		else:
			max_res = max(tupled_res, key=lambda x: x[0])
		print(max_res)
		results.append(max_res)

	# we need to sort first in order for the grouping operation to have effect.
	results = sorted(results, key=lambda x:x[1])
	sparsity_level_groups = groupby(results, key=lambda x:(x[1], x[3]))
	grouping = sparsity_level_groups
	colors = ["red", "blue", "green", "orange", "darkmagenta", "darkcyan"]
	markers = ["x", "o", "d", "p", "s", "D"]
	max_x = -1
	x_ticks = [0.0]
	for group_id, (grouping_key, group) in enumerate(grouping):
		print(group_id, grouping_key)
		y_axis = []
		x_axis = []
		# x_ticks = [0.0]
		num_params = []
		filenames = []

		for res in group:
			model_score = res[0]
			num_epochs_before_pruning = res[1]
			sparsity_level = res[2]
			num_epochs_after_pruning = res[3]
			model_params = res[4]
			purge_function = res[5]
			original_model_size = res[6]
			filename = res[7]

			y_axis.append(model_score)
			if progressive:
				sparsity_ = np.round(float(1 - model_params / original_model_size), 2)
				x_axis.append(sparsity_)
				x_ticks.append(sparsity_)
			else:
				x_axis.append(sparsity_level)
				x_ticks.append(sparsity_level)

			max_x = max(max_x, max(x_axis))
			num_params.append(model_params)
			filenames.append(filename)

		if "nopurge" in filename:
			line_label = "no-pruning: {}".format(num_epochs_after_pruning)
		else:
			line_label = "before-pruning: {}, after-pruning: {}".format(num_epochs_before_pruning,
																		num_epochs_after_pruning)
		print(group_id, line_label)
		for idx, (x, y, p, f) in enumerate(zip(x_axis, y_axis, num_params, filenames)):
			if idx == 0:
				ax.scatter(x, y, color=colors[group_id], marker=markers[group_id], label=line_label)
				# ax.scatter(x, y, label=line_label)
			else:
				ax.scatter(x, y, color=colors[group_id], marker=markers[group_id])
				# ax.scatter(x, y)
			# for bold text use: weight='heavy'
			ax.annotate(str(f'{p:,}'), xy=(x + 0.005, y + 0.0005), fontsize="xx-small",
						verticalalignment="bottom",
						horizontalalignment='left')

	# ax.set_xticks(set(sorted(x_ticks)))
	ax.set_xticks([t for t in set(x_ticks)])
	ax.set_xticklabels(ax.get_xticks(), rotation=90)
	ax.set_ylabel('Top-1 Accuracy')
	ax.set_xlabel('Sparsity Level')
	ax.set_ylim(0.0, 0.9)
	ax.plot([0.0, max_x], [0.89, 0.89], color="red", linestyle="-.", linewidth=1)

	fig.subplots_adjust(bottom=0.22)
	# fig.legend(title='Training Specs')
	fig.legend(title='Training Epochs', loc="lower center", bbox_to_anchor=(0.44, 0.01))

	return fig


def plot_federated_communication_cost_results(files, x_axis_federation_round=False, x_axis_transmission_cost=False,
									y_axis_secondary_params=False):

	fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
	if y_axis_secondary_params:
		ax2 = ax1.twinx()
		ax2.set_ylabel('Global Model Params', size=12)
		# ax2.set_ylim(0, 120000)
	# ax1.set_ylim(0.6, 0.8)
	bits_per_param = 32
	max_x = -1
	line_styles = ["solid", "--", "-."]

	for fidx, fname in enumerate(files):
		json_content = json.load(open(fname))
		merge_fn = json_content["federated_environment"]["merge_function"]
		number_of_learners = json_content["federated_environment"]["number_of_learners"]
		local_epochs = int(json_content["federated_environment"]["local_epochs_per_client"])
		prate = json_content["federated_environment"]["participation_rate"]
		sparsity = float(json_content["federated_environment"]["sparsity_level"])
		if "additional_specs" in json_content["federated_environment"]:
			sparsify_every_k_rounds = json_content["federated_environment"]["additional_specs"]["sparsify_every_k_round"]
		else:
			sparsify_every_k_rounds = 1
		start_purging_at_round = 0 if 'start_purging_at_round' not in json_content["federated_environment"] else json_content["federated_environment"]['start_purging_at_round']
		# print(start_purging_at_round)
		# sparsity = 1
		fine_tuning_epochs = float(json_content["federated_environment"]["fine_tuning_epochs"])
		federated_execution_results = json_content["federated_execution_results"]
		tupled_res = []
		incremental_comm_cost = 0
		for res in federated_execution_results:
			if 'local_models_total_params' in res:
				local_models_params_after_purging = res['local_models_total_params']
			elif 'local_models_total_params_after_purging' in res:
				local_models_params_after_purging = res['local_models_total_params_after_purging']
			total_participating_learners = len(local_models_params_after_purging)
			global_model_params = res['global_model_total_params']
			# communication cost per round in terms of model exchanges is: total_local_models + number_of_learners * global model
			# TODO We also need to add the mask when we use sparsification!!
			comm_cost = sum(local_models_params_after_purging) + res['global_model_total_params'] \
						* total_participating_learners
			comm_cost *= bits_per_param
			comm_cost /= 8e+6
			incremental_comm_cost += comm_cost
			tupled_res.append((res['global_model_test_score'], local_epochs, sparsity, fine_tuning_epochs,
							   incremental_comm_cost, global_model_params, start_purging_at_round))

		y1_axis = [res[0] for res in tupled_res]
		if x_axis_transmission_cost:
			ax1.set_xlabel('Transmission Cost (#MBs)', size=12)
			x_axis = [res[4] for res in tupled_res]
		if x_axis_federation_round:
			ax1.set_xlabel('Federation Round', size=12)
			x_axis = [x for x in range(len(y1_axis))]

		y1_axis = y1_axis[:200]
		x_axis = x_axis[:200]
		max_x = max(max(x_axis), max_x)

		# line_label = "le: {}, s:{}, ft: {}, wp: {}".format(local_epochs, sparsity, fine_tuning_epochs, start_purging_at_round)
		if 'MajorityVoting' in merge_fn:
			line_label = "le:{}, s:{}, wp:{}, f:{}, m:{}".format(local_epochs, sparsity, start_purging_at_round, sparsify_every_k_rounds, "MV")
		else:
			line_label = "le:{}, s:{}, wp:{}, f:{}, m:{}".format(local_epochs, sparsity, start_purging_at_round, sparsify_every_k_rounds, "FedAvg")
		avg_acc_last_10 = np.mean([t[0] for t in tupled_res[-10:]])
		avg_sparsification_last_10 = np.mean([1-t[-2]/118282 for t in tupled_res[-10:]])

		if "nopurging" in fname:
			ax1.plot(x_axis, y1_axis, label=line_label)
			y2_axis = [res[5] for res in tupled_res]
			y2_axis = y2_axis[:200]
			ax2.plot(x_axis, y2_axis, linestyle="--")

		# if avg_acc_last_10 > 0.74 and avg_sparsification_last_10 > 0.8:
		print(line_label, " max accuracy: {}, avg sparsification: {}, avg acc@sparsification: {}"
			  .format(max(y1_axis), avg_sparsification_last_10, avg_acc_last_10),
			  files[fidx])
		# print([(idx, y) for idx, y in enumerate(y1_axis)])
		ax1.plot(x_axis, y1_axis, label=line_label)

		if y_axis_secondary_params:
			y2_axis = [res[5] for res in tupled_res]
			y2_axis = y2_axis[:200]
			ax2.plot(x_axis, y2_axis, linestyle="--")

	# ax.set_xticks(sorted(set(x_ticks)))
	# ax1.plot([0.0, max_x], [0.7695000171661377, 0.7695000171661377], color="black", linestyle="-.", linewidth=2)
	ax1.set_ylabel('Test Top-1 Accuracy', size=12)
	fig.subplots_adjust(bottom=0.22)
	# fig.legend(title='Training Specs', loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=1, fontsize=12)
	fig.legend(loc="lower center", bbox_to_anchor=(0.44, 0.01), ncol=2, fontsize=12)

	return fig


def plot_federated_sparsified_results(files):

	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))

	results = []
	for f in files:
		json_content = json.load(open(f))
		number_of_learners = int(json_content["federated_environment"]["number_of_learners"])
		local_epochs = int(json_content["federated_environment"]["local_epochs_per_client"])
		prate = float(json_content["federated_environment"]["participation_rate"])
		sparsity = float(json_content["federated_environment"]["sparsity_level"])
		fine_tuning_epochs = float(json_content["federated_environment"]["fine_tuning_epochs"])
		purge_function_local = str(json_content["federated_environment"]["purge_function_local"])
		purge_function_global = str(json_content["federated_environment"]["purge_function_global"])
		merge_function = str(json_content["federated_environment"]["merge_function"])
		federated_execution_results = json_content["federated_execution_results"]
		tupled_res = [(
			res['global_model_test_score'],
			local_epochs,
			sparsity,
			fine_tuning_epochs,
			res['global_model_total_params'],
			purge_function_local,
			merge_function,
			f.split("/")[-3:]
		) for res in federated_execution_results[1:]]
		max_res = max(tupled_res, key=lambda x: x[0])
		results.append(max_res)

	results.sort(key=lambda x: x[2])
	fine_tuning_groups = groupby(results, key=lambda x: x[3])
	purge_merge_groups = groupby(results, key=lambda x: (x[5],x[6]))
	grouping = purge_merge_groups
	x_ticks = []
	colors = ["red", "blue", "green", "orange", "darkmagenta", "darkcyan"]
	markers = ["x", "o", "d", "p", "s", "D"]
	max_x = -1
	for group_id, (grouping_key, group) in enumerate(grouping):
		y_axis = []
		x_axis = []
		num_params = []
		filenames = []

		local_epochs = 0
		for res in group:
			y_axis.append(res[0])
			x_axis.append(res[2])
			x_ticks.append(res[2])
			max_x = max(max_x, max(x_axis))
			num_params.append(res[4])
			local_epochs = res[1]
			merge_function = res[5]
			purge_function = res[6]
			filenames.append(res[7])

		# line_label = "le: {}, ft: {}".format(local_epochs, grouping_key)
		line_label = grouping_key
		print(group_id, line_label)
		for idx, (x, y, p, f) in enumerate(zip(x_axis, y_axis, num_params, filenames)):
			if idx == 0:
				ax.scatter(x, y, color=colors[group_id], marker=markers[group_id])
			else:
				ax.scatter(x, y, color=colors[group_id], marker=markers[group_id])
			print(f,x,y)
			# for bold text use: weight='heavy'
			ax.annotate(str(f'{p:,}'), xy=(x + 0.005, y + 0.0005), fontsize="xx-small",
						verticalalignment="bottom",
						horizontalalignment='left')

	# ax.plot([0.0, max_x], [0.7695000171661377, 0.7695000171661377], color="red", linestyle="--")

	# ax.set_xticks(x_ticks)
	ax.set_ylabel('Top-1 Accuracy')
	ax.set_xlabel('Sparsity Level')

	fig.subplots_adjust(bottom=0.2)
	# fig.legend(title='Training Specs')
	fig.legend(title='Training Specs', loc="lower center", bbox_to_anchor=(0.5, -0.0))

	return fig


if __name__ == "__main__":

	files = []

	#### CENTRALIZED
	# files.extend(glob.glob("../logs/FashionMNIST/centralized/nopurge/*.json"))
	# files.extend(glob.glob("../logs/Cifar10/centralized/nopurge/ResNet/10blocks/*.json"))
	# files.extend(glob.glob("../logs/IMDB/centralized/nopurge/*.json"))

	# domain, main_dir = "FashionMNIST", "*/run1"
	# domain, main_dir = "Cifar10", "ResNet/10blocks/after_50epochs/batchsize_32"
	# domain, main_dir = "IMDB", "after_1epoch"

	# files.extend(glob.glob("../logs/{}/centralized/purge/one-shot-purging/{}/*05.*.json".format(domain, main_dir)))
	# files.extend(glob.glob("../logs/{}/centralized/purge/one-shot-purging/{}/*06.*.json".format(domain, main_dir)))
	# files.extend(glob.glob("../logs/{}/centralized/purge/one-shot-purging/{}/*07.*.json".format(domain, main_dir)))
	# files.extend(glob.glob("../logs/{}/centralized/purge/one-shot-purging/{}/*08.*.json".format(domain, main_dir)))
	# files.extend(glob.glob("../logs/{}/centralized/purge/one-shot-purging/{}/*085.*.json".format(domain, main_dir)))
	# files.extend(glob.glob("../logs/{}/centralized/purge/one-shot-purging/{}/*09.*.json".format(domain, main_dir)))
	# files.extend(glob.glob("../logs/{}/centralized/purge/one-shot-purging/{}/*095.*.json".format(domain, main_dir)))
	# files.extend(glob.glob("../logs/{}/centralized/purge/one-shot-purging/{}/*099.*.json".format(domain, main_dir)))

	# files.extend(glob.glob("../logs/{}/centralized/purge/progressive-purging/{}/*.json".format(domain, main_dir)))

	#### FEDERATED

	# files.extend(glob.glob("../logs/FashionMNIST/non-iid/classes2/fedpurgemerge/nopurging/vanillasgd_nopurging_weighted_avg_nnz/*learners_100*le_4*.json"))
	# files.extend(glob.glob("../logs/FashionMNIST/non-iid/classes2/fedpurgemerge/nopurging/vanillasgd_with_l1_nopurging_weighted_avg/l1_00001/*le_4*.json"))
	# files.extend(glob.glob("../logs/FashionMNIST/non-iid/classes2/fedpurgemerge/nopurging/vanillasgd_with_l2_nopurging_weighted_avg/l2_00001/*le_4*.json"))
	# files.extend(glob.glob("../logs/FashionMNIST/non-iid/classes2/fedpurgemerge/nopurging/vanillasgd_nopurging_weighted_avg_majority_voting_05/*le_4*.json"))

	# files.extend(glob.glob("../logs/FashionMNIST/non-iid/classes2/fedpurgemerge/nopurging/vanillasgd_nopurging_weighted_avg/*.learners_10.*.json"))
	# files.extend(glob.glob("../logs/FashionMNIST/non-iid/classes2/fedpurgemerge/pruning_baselines/snip/10learners/*.json"))
	# files.extend(glob.glob("../logs/FashionMNIST/non-iid/classes2/fedpurgemerge/progressive_pruning/purge_local_model/train_with_global_mask/vanillasgd_purge_local_model_random_nnz_merge_weighted_avg/100learners/*participation_01.*.json"))
	# files.extend(glob.glob("../logs/FashionMNIST/non-iid/classes2/fedpurgemerge/progressive_pruning/purge_local_model/train_with_global_mask/vanillasgd_purge_local_model_weight_layer_magnitude_nnz_merge_weighted_avg_majority_voting/10learners/*.json"))
	# files.extend(glob.glob("../logs/FashionMNIST/non-iid/classes2/fedpurgemerge/progressive_pruning/purge_local_model/train_with_global_mask/vanillasgd_purge_local_model_weight_model_magnitude_nnz_merge_weighted_avg_majority_voting/10learners/*.json"))
	# files.extend(glob.glob("../logs/FashionMNIST/non-iid/classes2/fedpurgemerge/progressive_pruning/purge_local_model/train_with_global_mask/sparsification_schedules/vanillasgd_purge_local_model_weight_model_magnitude_nnz_merge_weighted_avg_majority_voting/10learners/*.json"))
	# files.extend(glob.glob("../logs/FashionMNIST/*Fashion*.json"))
	# files.extend(glob.glob("../logs/FashionMNIST/non-iid/classes2/fedpurgemerge/*/*/*/vanillasgd_purge_local_model_weight_model_magnitude_nnz_merge_weighted_avg/10learners/FashionMNIST.rounds_200.learners_10.participation_1.le_4.compression_002.sparsificationround_20.finetuning_0.json"))
	# files.extend(glob.glob("../logs/FashionMNIST/non-iid/classes2/fedpurgemerge/*/*/*/*/10learners/*.json"))
	# files.extend(glob.glob("../logs/FashionMNIST/non-iid/classes2/fedpurgemerge/*/*/*/*/10learners/*.json"))
	files.extend(glob.glob("../logs/FashionMNIST/non-iid/classes2/fedpurgemerge/progressive_pruning/purge_local_model/train_with_global_mask/sparsification_schedules/*/10learners/*compression_004*.json"))

	# files.extend(glob.glob("../logs/IMDB/iid/fedpurgemerge/nopurging/*.json"))
	# files.extend(glob.glob("../logs/IMDB/iid/fedpurgemerge/progressive_pruning/train_with_global_mask/vanillasgd_purge_local_model_weight_model_magnitude_nnz_merge_weighted_avg_majority_voting/10learners/*.json"))
	# files.extend(glob.glob("../logs/IMDB/iid/fedpurgemerge/purge_weight_magnitude_local_merge_weighted_avg/*.json"))

	# files = sorted(files, key=lambda f: float(json.load(open(f))["federated_environment"]["sparsity_level"]))

	# fig = plot_centralized_sparsified_results(files, progressive=False)
	# fig = plot_centralized_epoch_results(files,
	# 									 x_axis_epoch=True,
	# 									 y_axis_secondary_params=True)

	# fig = plot_federated_sparsified_results(files)
	fig = plot_federated_communication_cost_results(files,
													x_axis_federation_round=True,
													x_axis_transmission_cost=False,
													y_axis_secondary_params=True)

	file_out = "{}.pdf".format("PurgeMerge")
	pdfpages1 = PdfPages(file_out)
	pdfpages1.savefig(figure=fig, bbox_inches='tight')
	pdfpages1.close()
	plt.show()

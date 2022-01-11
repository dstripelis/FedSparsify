import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap


def plot_model_binary_masks(model_mask_npz_fp=None):

	model_weights_mask = np.load(model_mask_npz_fp)
	flatten_mask = np.concatenate([w_val.flatten() for w_val in model_weights_mask.values()])
	non_zeros_num = len(flatten_mask[flatten_mask != 0.0])
	zeros_num = len(flatten_mask[flatten_mask == 0.0])
	print("Z:", zeros_num)
	print("NNZ:", non_zeros_num)

	hibernation_color = "black"
	# activation_color = "#FFA500"
	activation_color = "forestgreen"
	colors = [hibernation_color, activation_color]
	if zeros_num == 0:
		flatten_mask = np.insert(flatten_mask, 0, 0)
	print(flatten_mask)
	cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))
	# Set up the matplotlib figure
	f, ax = plt.subplots(figsize=(12, 5))
	ax.set_title("Model Binary Mask")
	heatmap_plot = sns.heatmap([flatten_mask], cbar_kws={"shrink": .5}, cmap=cmap)
	# heatmap_plot = sns.heatmap([flatten_mask], cbar=False, annot=False, cmap='rocket', linewidths=1.3, vmin=0, vmax=1)
	colorbar = heatmap_plot.collections[0].colorbar
	colorbar.set_ticks([0.25, 0.75])
	colorbar.set_ticklabels(['0', '1'])
	# heatmap_plot.set(xticks=[])
	heatmap_plot.set(yticks=[])
	heatmap_plot.set(xlabel="Flattened Model Params Positions")
	fig = heatmap_plot.get_figure()
	fig.subplots_adjust(bottom=0.4)

	return fig


def plot_model_values_distribution(model_npz_fp, model_mask_npz_fp=None):
	print(model_npz_fp)
	print(model_mask_npz_fp)
	model_weights = np.load(model_npz_fp)
	flattened_model = np.concatenate([w_val.flatten() for w_val in model_weights.values()])
	non_zeros_num = len(flattened_model[flattened_model != 0.0])
	zeros_num = len(flattened_model[flattened_model == 0.0])
	print("Z:", zeros_num)
	print("NNZ:", non_zeros_num)

	flatten_mask = [1 for v in flattened_model]
	if model_mask_npz_fp is not None:
		model_weights_mask = np.load(model_mask_npz_fp)
		flatten_mask = np.concatenate([w_val.flatten() for w_val in model_weights_mask.values()])

	# max_pruned_value = -np.inf
	# min_pruned_value = np.inf
	# for val, mask in zip(flattened_model, flatten_mask):
	# 	if mask == 1.0:
	# 		max_pruned_value = max(val, max_pruned_value)
	# 		min_pruned_value = min(val, min_pruned_value)
	# print(max_pruned_value, min_pruned_value)

	final_model = np.multiply(flattened_model, flatten_mask)
	# final_model = final_model[final_model != 0.0]

	distribution_plot = sns.histplot(final_model[final_model > 0.0], color="tab:blue")
	distribution_plot = sns.histplot(final_model[final_model < 0.0], color="tab:blue")
	# distribution_plot = sns.histplot(final_model[final_model == 0.0], color="white")
	# distribution_plot = sns.histplot(final_model)
	# distribution_plot = sns.distplot(final_model, hist=True,
	# 			 bins=int(180/5), color = 'darkblue',
	# 			 hist_kws={'edgecolor':'black'},
	# 			 kde_kws={'linewidth': 1})
	fig = distribution_plot.get_figure()
	return fig


if __name__ == "__main__":

	files = []

	# === No Purging ===
	# fig = plot_model_values_distribution(
	# 	model_npz_fp="/data/stripeli/projectmetis/simulatedFL/npzarrays/FashionMNIST/centralized/nopurge/FashionMNIST.centralized.epochs_100.lr_002/global_model_federation_round_100.npz",
	# 	model_mask_npz_fp="/data/stripeli/projectmetis/simulatedFL/npzarrays/FashionMNIST/centralized/nopurge/FashionMNIST.centralized.epochs_100.lr_002/global_model_masks_federation_round_100.npz"
	# )
	# fig = plot_model_binary_masks(
	# 	model_mask_npz_fp="/data/stripeli/projectmetis/simulatedFL/npzarrays/FashionMNIST/centralized/nopurge/FashionMNIST.centralized.epochs_100.lr_002/global_model_masks_federation_round_100.npz"
	# )

	# === Purging ===
	# fig = plot_model_values_distribution(
	# 	model_npz_fp="/data/stripeli/projectmetis/simulatedFL/npzarrays/FashionMNIST/centralized/purge/one_shot_purging/after_10epochs/FashionMNIST.rounds_100.learners_1.participation_1.le_1.compression_07.sparsificationround_10.finetuning_0/global_model_federation_round_100.npz",
	# 	model_mask_npz_fp="/data/stripeli/projectmetis/simulatedFL/npzarrays/FashionMNIST/centralized/purge/one_shot_purging/after_10epochs/FashionMNIST.rounds_100.learners_1.participation_1.le_1.compression_07.sparsificationround_10.finetuning_0/global_model_masks_federation_round_100.npz"
	# )
	fig = plot_model_binary_masks(
		model_mask_npz_fp="/data/stripeli/projectmetis/simulatedFL/npzarrays/FashionMNIST/centralized/purge/one_shot_purging/after_90epochs/FashionMNIST.rounds_100.learners_1.participation_1.le_1.compression_099.sparsificationround_90.finetuning_0/global_model_masks_federation_round_100.npz"
	)
	# fig = plot_model_binary_masks(
	# 	model_mask_npz_fp="/data/stripeli/projectmetis/simulatedFL/npzarrays/FashionMNIST/centralized/purge/progressive_purging/after_1epoch/FashionMNIST.rounds_100.learners_1.participation_1.le_1.compression_002.sparsificationround_1.finetuning_0/global_model_masks_federation_round_100.npz"
	# )

	file_out = "{}.png".format("PurgeMerge")
	plt.tight_layout()
	plt.savefig(file_out)
	# file_out = "{}.pdf".format("PurgeMerge")
	# pdfpages1 = PdfPages(file_out)
	# pdfpages1.savefig(figure=fig, bbox_inches='tight')
	# pdfpages1.close()
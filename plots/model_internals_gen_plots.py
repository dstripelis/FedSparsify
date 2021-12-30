import numpy as np
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages



def plot_model_distribution(model_npz_fp, model_mask_npz_fp=None):
	print(model_npz_fp)
	print(model_mask_npz_fp)
	model_weights = np.load(model_npz_fp)
	flattened_model = np.concatenate([w_val.flatten() for w_val in model_weights.values()])

	flatten_mask = [1 for v in flattened_model]
	if model_mask_npz_fp is not None:
		model_weights_mask = np.load(model_mask_npz_fp)
		flatten_mask = np.concatenate([w_val.flatten() for w_val in model_weights.values()])

	final_model = np.multiply(flattened_model, flatten_mask)
	# final_model = [x!=0.0 for x in np.multiply(flattened_model, flatten_mask)]

	distribution_plot = sns.distplot(final_model, hist=True, kde=True,
				 bins=int(180/5), color = 'darkblue',
				 hist_kws={'edgecolor':'black'},
				 kde_kws={'linewidth': 4})
	fig = distribution_plot.get_figure()
	return fig


if __name__ == "__main__":

	files = []

	# fig = plot_model_distribution(
	# 	model_npz_fp="/data/stripeli/projectmetis/simulatedFL/npzarrays/FashionMNIST/centralized/nopurge/FashionMNIST.centralized.epochs_100.lr_002/global_model_federation_round_100.npz",
	# 	model_mask_npz_fp="/data/stripeli/projectmetis/simulatedFL/npzarrays/FashionMNIST/centralized/nopurge/FashionMNIST.centralized.epochs_100.lr_002/global_model_masks_federation_round_100.npz"
	# )
	fig = plot_model_distribution(
		model_npz_fp="/data/stripeli/projectmetis/simulatedFL/npzarrays/FashionMNIST/centralized/purge/one_shot_purging/after_10epochs/FashionMNIST.rounds_100.learners_1.participation_1.le_1.compression_08.sparsificationround_10.finetuning_0/global_model_federation_round_100.npz",
		model_mask_npz_fp="/data/stripeli/projectmetis/simulatedFL/npzarrays/FashionMNIST/centralized/purge/one_shot_purging/after_10epochs/FashionMNIST.rounds_100.learners_1.participation_1.le_1.compression_08.sparsificationround_10.finetuning_0/global_model_masks_federation_round_100.npz"
	)
	file_out = "{}.pdf".format("PurgeMerge")
	pdfpages1 = PdfPages(file_out)
	pdfpages1.savefig(figure=fig, bbox_inches='tight')
	pdfpages1.close()
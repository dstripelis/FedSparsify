"""
For this environment to work we need to activate the appropriate python interpreter:
	conda activate /data/stripeli/condaenvmlbench

Once the .onnx models have been saved/generated we can run the following command to
benchmark inference time of sparse vs dense models:
 	deepsparse.benchmark -b 128 -t 60 -w 10 --export_path /tmp/cifar10_resnet.0percent.inference_eval.json /tmp/cifar10_resnet.0percent.onnx
 	see also: https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/benchmark
"""


from simulatedFL.models.brainage3d_cnn import BrainAge3DCNN
from simulatedFL.models.imdb_lstm import IMDB_LSTM
from simulatedFL.models.cifar.cifar_cnn import CifarCNN
from simulatedFL.models.cifar.cifar_resnet_v2 import ResNetCifar10

import simulatedFL.models as models
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

import os
import tf2onnx
import zipfile

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.

  zipped_file = file + ".gzip"
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  bytes_size = os.path.getsize(zipped_file)
  megabytes_size = 0.000001 * bytes_size
  return (bytes_size, megabytes_size)


def measure_model_size(model, output_filepath, npz_filepath=None, convert_to_onnx=True, input_spec=None):

	if npz_filepath is not None:
		print("Analyzing:")
		print(npz_filepath)
		npz_file = np.load(npz_filepath)
		weights = [v for w, v in npz_file.items()]
		model.set_weights(weights)
	print("Non-Zero Params:")
	print(sum([np.count_nonzero(w) for w in model.get_weights()]))

	model = tfmot.sparsity.keras.strip_pruning(model)
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
	tflite_model = converter.convert()

	# Save the TF Lite model as file
	tflite_output_filepath = output_filepath + ".tflite"
	f = open(tflite_output_filepath, "wb")
	f.write(tflite_model)
	f.close()

	model_size = get_gzipped_model_size(tflite_output_filepath)

	if convert_to_onnx:
		"""
		You convert tflite models via command line:
		python -m tf2onnx.convert --opset 16 --tflite tflite--file --output model.onnx
		"""
		onnx_output_filepath = output_filepath + ".onnx"
		model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_spec,
													opset=13, output_path=onnx_output_filepath)
		output_names = [n.name for n in model_proto.graph.output]
		print(output_names)

	return model_size


if __name__ == "__main__":

	cifar10_cnn, cifar10_resnet, imdb_bilstm, brainage_cnn = False, False, True, False
	if cifar10_cnn:
		model = CifarCNN(cifar_10=True).get_model()
		input_spec = (tf.TensorSpec((None, 32, 32, 3), tf.float32, name="input"),)
		sparsities = [80, 85, 90, 95, 99]
	if cifar10_resnet:
		model = ResNetCifar10(num_layers=56).get_model()
		input_spec = (tf.TensorSpec((None, 32, 32, 3), tf.float32, name="input"),)
		sparsities = [80, 85, 90, 95, 99]
	if imdb_bilstm:
		model = IMDB_LSTM().get_model()
		input_spec = (tf.TensorSpec((16, 64), tf.float32, name="input"),)
		sparsities = [85, 90, 95, 99]
	if brainage_cnn:
		model = BrainAge3DCNN().get_model(training_batch_norm=False)
		input_spec = (tf.TensorSpec((None, 91, 109, 91, 1), tf.float32, name="input"),)
		sparsities = [80, 85, 90, 95, 99]

	all_model_sizes = dict()
	# Dense Model
	model_size = measure_model_size(model, '/tmp/imdb_bilstm.0percent', npz_filepath=None, input_spec=input_spec)
	all_model_sizes["sparsity:0"] = model_size
	print(model_size, " (bytes, MBs)")
	print("\n\n")

	# Sparse Models
	for sparsity in sparsities:
		if sparsity % 10 == 0:
			# npz_filepath = "/data/stripeli/projectmetis/simulatedFL/npzarrays/Cifar10/Cifar10.CNN.FedSparsifyRandom.NonIID.rounds_200.learners_10.participation_1.le_4.compression_0{}.sparsificationround_1.sparsifyevery_1rounds.finetuning_0/global_model_federation_round_200.npz".format(int(sparsity/10))
			# npz_filepath = "/data/stripeli/projectmetis/simulatedFL/npzarrays/Cifar10/Cifar10.ResNet.FedSparsifyRandom.NonIID.rounds_100.learners_10.participation_1.le_4.compression_0{}.sparsificationround_1.sparsifyevery_1rounds.finetuning_0/global_model_federation_round_100.npz".format(int(sparsity/10))
			npz_filepath = "/data/stripeli/projectmetis/simulatedFL/npzarrays/IMDB/IMDB.FedSparsifyGlobal.IID.0{}.rounds_200.learners_10.participation_1.le_4.compression_0{}.sparsificationround_1.sparsifyevery_1rounds.finetuning_0/global_model_federation_round_200.npz".format(int(sparsity/10), int(sparsity/10))
			# npz_filepath = "/data/stripeli/projectmetis/simulatedFL/npzarrays/BrainAge/BrainAge.UniformNonIID.rounds_40.learners_8.participation_1.le_4.compression_09.sparsificationround_1.sparsifyevery_1rounds.finetuning_0/global_model_federation_round_40.npz"
		else:
			# npz_filepath = "/data/stripeli/projectmetis/simulatedFL/npzarrays/Cifar10/Cifar10.CNN.FedSparsifyRandom.NonIID.rounds_200.learners_10.participation_1.le_4.compression_0{}.sparsificationround_1.sparsifyevery_1rounds.finetuning_0/global_model_federation_round_200.npz".format(sparsity)
			# npz_filepath = "/data/stripeli/projectmetis/simulatedFL/npzarrays/Cifar10/Cifar10.ResNet.FedSparsifyRandom.NonIID.rounds_100.learners_10.participation_1.le_4.compression_0{}.sparsificationround_1.sparsifyevery_1rounds.finetuning_0/global_model_federation_round_100.npz".format(sparsity)
			npz_filepath = "/data/stripeli/projectmetis/simulatedFL/npzarrays/IMDB/IMDB.FedSparsifyGlobal.IID.0{}.rounds_200.learners_10.participation_1.le_4.compression_0{}.sparsificationround_1.sparsifyevery_1rounds.finetuning_0/global_model_federation_round_200.npz".format(sparsity, sparsity)
			# npz_filepath = "/data/stripeli/projectmetis/simulatedFL/npzarrays/BrainAge/BrainAge.UniformNonIID.rounds_40.learners_8.participation_1.le_4.compression_0{}.sparsificationround_1.sparsifyevery_1rounds.finetuning_0/global_model_federation_round_40.npz"
		model_size = measure_model_size(model, '/tmp/imdb_bilstm.{}percent'.format(sparsity), npz_filepath=npz_filepath, input_spec=input_spec)
		all_model_sizes["sparsity:{}".format(sparsity)] = model_size
		print(model_size, " (bytes, MBs)")
		print("\n\n")

	for sparsity, model_size in all_model_sizes.items():
		print(sparsity, model_size)
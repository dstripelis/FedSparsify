from simulatedFL.models.brainage3d_cnn import BrainAge3DCNN

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

import os
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


# Dense Model
npzfile = np.load("/data/stripeli/projectmetis/simulatedFL/npzarrays/BrainAge/BrainAge.UniformNonIID.rounds_40.learners_8.participation_1.le_4.compression_099.sparsificationround_1.sparsifyevery_1rounds.finetuning_0/global_model_federation_round_1.npz")
model = BrainAge3DCNN().get_model(training_batch_norm=False)
weights = [v for w, v in npzfile.items()]
model.set_weights(weights)

print("Non-Zero Params:")
print(sum([np.count_nonzero(w) for w in model.get_weights()]))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

output_file = '/tmp/brainage.0percent.tflite'
# Save the TF Lite model as file
f = open(output_file, "wb")
f.write(tflite_model)
f.close()

print(get_gzipped_model_size(output_file), " (bytes, MBs)")
print("\n\n")

# Sparse Models
for sparsity in [85, 90, 95, 99]:
	if sparsity == 90:
		npzfile = np.load("/data/stripeli/projectmetis/simulatedFL/npzarrays/BrainAge/BrainAge.UniformNonIID.rounds_40.learners_8.participation_1.le_4.compression_09.sparsificationround_1.sparsifyevery_1rounds.finetuning_0/global_model_federation_round_40.npz")
	else:
		npzfile = np.load("/data/stripeli/projectmetis/simulatedFL/npzarrays/BrainAge/BrainAge.UniformNonIID.rounds_40.learners_8.participation_1.le_4.compression_0{}.sparsificationround_1.sparsifyevery_1rounds.finetuning_0/global_model_federation_round_40.npz".format(sparsity))

	model = BrainAge3DCNN().get_model(training_batch_norm=False)
	weights = [v for w, v in npzfile.items()]
	model.set_weights(weights)

	# model = tfmot.sparsity.keras.prune_low_magnitude(model)
	# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
	# 			  loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE),
	# 			  metrics=["mae"])
	# model.set_weights(weights)
	model = tfmot.sparsity.keras.strip_pruning(model)

	print("Non-Zero Params:")
	print(sum([np.count_nonzero(w) for w in model.get_weights()]))

	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
	tflite_model = converter.convert()

	output_file = '/tmp/brainage.{}percent.tflite'.format(sparsity)
	# Save the TF Lite model as file
	f = open(output_file, "wb")
	f.write(tflite_model)
	f.close()

	print(get_gzipped_model_size(output_file), " (bytes, MBs)")
	print("\n\n")




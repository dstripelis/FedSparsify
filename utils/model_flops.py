import numpy as np

def get_flops(concrete_func):
	import tensorflow as tf
	from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

	frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)

	with tf.Graph().as_default() as graph:
		tf.graph_util.import_graph_def(graph_def, name='')

		run_meta = tf.compat.v1.RunMetadata()
		opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
		flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

		return flops.total_float_ops


import tensorflow as tf
from simulatedFL.models.fashion_mnist_fc import FashionMnistModel
from simulatedFL.models.cifar.cifar_cnn import CifarCNN
from simulatedFL.models.imdb_lstm import IMDB_LSTM

model = FashionMnistModel().get_model()  # total flops per example: 236,348
# model = CifarCNN(cifar_10=True).get_model()  # total flops per example: 232,458,812
# model = CifarCNN(cifar_100=True).get_model()  # total flops per example: 232,551,512
# model = IMDB_LSTM().get_model()  # total flops per example: 131,088

new_weights = [w for w in model.get_weights()]
w_shape = new_weights[0].shape
new_weights[0] = np.zeros(w_shape)
model.set_weights(new_weights)
for w in model.get_weights():
	print(w)

concrete = tf.function(lambda inputs: model(inputs))
concrete_func = concrete.get_concrete_function(
    [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])

print("total flops:", get_flops(concrete_func))
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

import numpy as np
# np.random.seed(1990)

import random
# random.seed(1990)

import tensorflow as tf
# tf.random.set_seed(1990)

from simulatedFL.models.cifar.cifar_vgg import CifarVGG

def data_augmentation_fn(*image_label_tuple):
	image, label = image_label_tuple
	# image = tf.pad(image, [[4, 4],
	# 					   [4, 4], [0, 0]])
	padding = 4
	image_size = 32
	target_size = image_size + padding * 2
	image = tf.image.pad_to_bounding_box(image, padding, padding, target_size, target_size)
	image = tf.image.random_flip_left_right(image, seed=1990)
	image = tf.image.random_crop(image, [32, 32, 3])
	return image, label

def main(vgg_type, with_nesterov, learning_rate):

	num_classes = 100
	batch_size = 128
	epochs = 100

	"""Load the data."""
	if num_classes == 10:
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
	else:
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
	input_shape = [None] + list(x_train.shape[1:])

	x_train = x_train.astype('float32') / 255
	# The following requires to set the loss function to CategoricalCrossEntropy().
	# y_train = tf.keras.utils.to_categorical(y_train, num_classes)
	x_test = x_test.astype('float32') / 255
	# The following requires to set the loss function to CategoricalCrossEntropy().
	# y_test = tf.keras.utils.to_categorical(y_test, num_classes)

	# normalize data
	train_channel_mean = np.mean(x_train, axis=(0, 1, 2))
	train_channel_std = np.std(x_train, axis=(0, 1, 2))
	test_channel_mean = np.mean(x_train, axis=(0, 1, 2))
	test_channel_std = np.std(x_train, axis=(0, 1, 2))

	for i in range(3):
		x_train[:, :, :, i] = (x_train[:, :, :, i] - train_channel_mean[i]) / train_channel_std[i]
		x_test[:, :, :, i] = (x_test[:, :, :, i] - test_channel_mean[i]) / test_channel_std[i]

	train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	train_dataset = \
		train_dataset.map(data_augmentation_fn).shuffle(buffer_size=5000, seed=1990).batch(batch_size)

	# Train
	num_samples = len(y_train)
	decay_steps = int(epochs * num_samples / batch_size)
	model = CifarVGG(input_shape, vgg_type, with_nesterov, learning_rate, num_classes=num_classes)
	model1 = model.get_model()
	model1.summary()
	model1.fit(train_dataset,
			   batch_size=batch_size,
			   epochs=100,
			   validation_freq=1,
			   validation_data=(x_test, y_test),
			   verbose=1)

	# Evaluate
	score = model.evaluate(x_test, y_test, verbose=1)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])


if __name__ == "__main__":
	# print("CIFAR-10")
	# for vgg_type in ("VGG-16",):
	# 	for with_nesterov in (True, False):
	# 		for learning_rate in (1e-1, 1e-2, 1e-3, 3e-2, 3e-3):
	# 			print("RUNNING, VGG: {}, Nesterov: {}, LearningRate: {}".format(vgg_type, with_nesterov, learning_rate))
	# 			main(vgg_type, with_nesterov, learning_rate)
	# 			print("\n\n\n*************\n\n\n")
	print("CIFAR-100")
	for vgg_type in ("VGG-16",):
		for with_nesterov in (True,):
			for learning_rate in (1e-2,):
				print("RUNNING, VGG: {}, Nesterov: {}, LearningRate: {}".format(vgg_type, with_nesterov, learning_rate))
				main(vgg_type, with_nesterov, learning_rate)
				print("\n\n\n*************\n\n\n")

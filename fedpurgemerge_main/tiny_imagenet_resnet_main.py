import six
import os

from simulatedFL.data.tinyimagenet.process_raw_files import ProcessImagenetRawData
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Activation, Dense, Flatten)
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Add)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

epochs = 100
num_classes = 200
batch_size = 32
nb_epoch = 5

# Load images
path = '/data/stripeli/projectmetis/simulatedFL/data/tiny-imagenet-200'
x_train, y_train, x_test, y_test = ProcessImagenetRawData(path).load_images(num_classes)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# input image dimensions
img_rows, img_cols = 64, 64
# The images are RGB
img_channels = 3

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# normalize data
train_channel_mean = np.mean(x_train, axis=(0, 1, 2))
train_channel_std = np.std(x_train, axis=(0, 1, 2))
test_channel_mean = np.mean(x_train, axis=(0, 1, 2))
test_channel_std = np.std(x_train, axis=(0, 1, 2))

for i in range(3):
	x_train[:, :, :, i] = (x_train[:, :, :, i] - train_channel_mean[i]) / train_channel_std[i]
	x_test[:, :, :, i] = (x_test[:, :, :, i] - test_channel_mean[i]) / test_channel_std[i]


def data_augmentation_fn(*image_label_tuple):
	image, label = image_label_tuple
	# image = tf.pad(image, [[4, 4],
	# 					   [4, 4], [0, 0]])
	padding = 4
	image_size = 64
	target_size = image_size + padding * 2
	image = tf.image.pad_to_bounding_box(image, padding, padding, target_size, target_size)
	image = tf.image.random_flip_left_right(image, seed=1990)
	image = tf.image.random_crop(image, [64, 64, 3])
	return image, label


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = \
	train_dataset.map(data_augmentation_fn).shuffle(buffer_size=5000, seed=1990).batch(batch_size)

# num_samples = len(y_train)
# decay_steps = int(epochs * num_samples / batch_size)
# learning_rate_fn = tf.keras.experimental.CosineDecay(1e-1, decay_steps=decay_steps)
for learning_rate in (1e-1, 1e-2, 1e-3):
	print("Learning Rate: {}".format(learning_rate))
	resnet50 = tf.keras.applications.resnet50.ResNet50(
		include_top=True,
		weights=None,
		input_tensor=None,
		input_shape=(64, 64, 3),
		pooling=None,
		classes=num_classes)
	model = resnet50
	optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
	model.compile(loss='categorical_crossentropy',
				  optimizer=optimizer,
				  metrics=['accuracy'])

	model.fit(train_dataset,
			  batch_size=batch_size,
			  epochs=epochs,
			  validation_freq=nb_epoch,
			  validation_data=(x_test, y_test),
			  verbose=1)

	score = model.evaluate(x_test, y_test, verbose=1)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
	print("\n\n\n*************\n\n\n")

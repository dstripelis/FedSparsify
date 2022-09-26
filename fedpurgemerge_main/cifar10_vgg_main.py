from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

import os
import random
import tensorflow as tf
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

np.random.seed(1990)
random.seed(1990)
tf.random.set_seed(1990)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

num_classes = 10
batch_size = 32
nb_epoch = 5

# Load images
"""Load the data."""
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
input_shape = x_train.shape[1:]

x_test = x_test.astype('float32') / 255
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, num_classes)

# normalize data
test_channel_mean = np.mean(x_train, axis=(0, 1, 2))
test_channel_std = np.std(x_train, axis=(0, 1, 2))
train_channel_mean = np.mean(x_train, axis=(0, 1, 2))
train_channel_std = np.std(x_train, axis=(0, 1, 2))

for i in range(3):
	x_test[:, :, :, i] = (x_test[:, :, :, i] - test_channel_mean[i]) / test_channel_std[i]
	x_train[:, :, :, i] = (x_train[:, :, :, i] - train_channel_mean[i]) / train_channel_std[i]

vgg19 = tf.keras.applications.VGG19(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(32, 32, 3),
    pooling=None,
    classes=10,
    classifier_activation="softmax",
)

model = Sequential()
for layer in vgg19.layers:
    model.add(layer)

model.compile(loss="categorical_crossentropy",
			  optimizer=SGD(lr=0.001, momentum=0.9, nesterov=False),
			  metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,
		  batch_size=batch_size,
		  epochs=100,
		  validation_freq=nb_epoch,
		  validation_data=(x_test, y_test),
		  verbose=1)

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

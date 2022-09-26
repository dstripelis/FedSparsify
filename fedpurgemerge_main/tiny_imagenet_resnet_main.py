import six
import os

from simulatedFL.data.tinyimagenet.process_raw_files import ProcessImagenetRawData
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Activation, Dense, Flatten)
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Add)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

num_classes = 200
batch_size = 32
nb_epoch = 10

resnet50 = tf.keras.applications.resnet50.ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(64, 64, 3),
    pooling=None,
    classes=num_classes)



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

# subtract mean and normalize
x_train -= np.mean(x_train, axis=0)
x_test -= np.mean(x_test, axis=0)
x_train /= 128.
x_test /= 128.

model = resnet50
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=100, validation_freq=nb_epoch,
		  verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

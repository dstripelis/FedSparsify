import os
import tensorflow.keras.utils as keras_utils
import numpy as np

from simulatedFL.data.tinyimagenet.process_raw_files import ProcessImagenetRawData

# Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam


os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# Params
# loss_functions = ['categorical_crossentropy', 'squared_hinge', 'hinge']
num_classes = 200
batch_size = 32
nb_epoch = 5

# Load images
path = '/data/stripeli/projectmetis/simulatedFL/data/tiny-imagenet-200'
X_train, Y_train, X_test, Y_test = ProcessImagenetRawData(path).load_images(num_classes, shuffle_train=True)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

num_samples = len(X_train)

# input image dimensions
num_channels, img_rows, img_cols = X_train.shape[1], X_train.shape[2], X_train.shape[3]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_train /= 128.
X_test /= 128.

# convert class vectors to binary class matrices
Y_train = keras_utils.to_categorical(Y_train, num_classes)
Y_test = keras_utils.to_categorical(Y_test, num_classes)


model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3, 64, 64)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', ))
model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3,
						activation='relu',
						kernel_regularizer=l1_l2(1e-7, 1e-7),
						activity_regularizer=l1_l2(1e-7, 1e-7)))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3,
						activation='relu',
						kernel_regularizer=l1_l2(1e-6, 1e-6),
						activity_regularizer=l1_l2(1e-6, 1e-6)))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3,
						activation='relu',
						kernel_regularizer=l1_l2(1e-5, 1e-5),
						activity_regularizer=l1_l2(1e-5, 1e-5)))
model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

model.add(Flatten())
model.add(Dense(4096,
				activation='relu',
				kernel_regularizer=l1_l2(1e-4, 1e-4),
				activity_regularizer=l1_l2(1e-5, 1e-5)))
model.add(Dropout(0.75))
model.add(Dense(4096,
				activation='relu',
				kernel_regularizer=l1_l2(1e-4, 1e-4),
				activity_regularizer=l1_l2(1e-5, 1e-5)))
model.add(Dropout(0.75))
model.add(Dense(200, activation='softmax'))

# if loss_function == 'categorical_crossentropy':
# 	# affine layer w/ softmax activation added
# 	model.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2(1e-5)))
# 	sgd = SGD(lr=0.05, decay=1e-5, momentum=0.9, nesterov=True)
# else:
# 	if loss_function == 'hinge':
# 		sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
# 	else:
# 		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# 	model.add(Dense(num_classes, kernel_regularizer=l2(1e-5)))

model.compile(loss="categorical_crossentropy",
			  optimizer=SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),
			  metrics=['accuracy'])

model.summary()

# model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True, save_to_dir='./datagen/', save_prefix='datagen-',save_format='png'), # To save the images created by the generator
# samples_per_epoch=num_samples, nb_epoch=nb_epoch,
# verbose=1, validation_data=(X_test,Y_test),
# callbacks=[Plotter(show_regressions=False, save_to_filepath=fpath, show_plot_window=False)])

model.fit(X_train, Y_train,
		  batch_size=batch_size,
		  epochs=100,
		  validation_freq=nb_epoch,
		  validation_data=(X_test, Y_test),
		  verbose=1)

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

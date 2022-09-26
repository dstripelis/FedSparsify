# Sys
import os
import numpy as np
from PIL import Image

class ProcessImagenetRawData:

	def __init__(self, path):
		self.path = path

	def __get_annotations_map(self):
		val_annotations_path = self.path + '/val/val_annotations.txt'
		val_annotations_file = open(val_annotations_path, 'r')
		val_annotations_contents = val_annotations_file.read()
		val_annotations = {}

		for line in val_annotations_contents.splitlines():
			pieces = line.strip().split()
			val_annotations[pieces[0]] = pieces[1]

		return val_annotations


	def load_images(self, num_classes, shuffle_train=True):
		# Load images

		print('Loading ' + str(num_classes) + ' classes')

		X_train = np.zeros([num_classes * 500, 3, 64, 64], dtype='uint8')
		y_train = np.zeros([num_classes * 500], dtype='uint8')

		trainPath = self.path + '/train'

		print('loading training images...')

		i = 0
		j = 0
		annotations = {}
		for sChild in os.listdir(trainPath):
			sChildPath = os.path.join(os.path.join(trainPath, sChild), 'images')
			annotations[sChild] = j
			for c in os.listdir(sChildPath):
				X = np.array(Image.open(os.path.join(sChildPath, c)))
				if len(np.shape(X)) == 2:
					X_train[i] = np.array([X, X, X])
				else:
					X_train[i] = np.transpose(X, (2, 0, 1))
				y_train[i] = j
				i += 1
			j += 1
			if (j >= num_classes):
				break

		X_train = np.transpose(X_train, (0, 2, 3, 1))

		print('finished loading training images')

		val_annotations_map = self.__get_annotations_map()

		X_test = np.zeros([num_classes * 50, 3, 64, 64], dtype='uint8')
		y_test = np.zeros([num_classes * 50], dtype='uint8')

		print('loading test images...')

		i = 0
		testPath = self.path + '/val/images'
		for sChild in os.listdir(testPath):
			if val_annotations_map[sChild] in annotations.keys():
				sChildPath = os.path.join(testPath, sChild)
				X = np.array(Image.open(sChildPath))
				if len(np.shape(X)) == 2:
					X_test[i] = np.array([X, X, X])
				else:
					X_test[i] = np.transpose(X, (2, 0, 1))
				y_test[i] = annotations[val_annotations_map[sChild]]
				i += 1
			else:
				pass
		X_test = np.transpose(X_test, (0, 2, 3, 1))

		print('finished loading test images')

		if shuffle_train:
			size = len(X_train)
			train_idx = np.arange(size)
			np.random.shuffle(train_idx)
			X_train, y_train = X_train[train_idx], y_train[train_idx]

		return X_train, y_train, X_test, y_test

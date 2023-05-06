from models.model import Model
from utils.optimizers.fed_prox import FedProx
from tensorflow.keras import layers

import tensorflow as tf

'''
VGG11/13/16/19 in TensorFlow2.
Reference:
[1] Simonyan, Karen, and Andrew Zisserman. 
    "Very deep convolutional networks for large-scale image recognition." 
    arXiv preprint arXiv:1409.1556 (2014).
'''

class VGG__(tf.keras.Model):

	def __init__(self, vgg_name, num_classes):
		super(VGG__, self).__init__()
		self.config = {
			'VGG-11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
			'VGG-13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
			'VGG-16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
			'VGG-19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
		}
		self.conv = self._make_layers(self.config[vgg_name])
		self.flatten = layers.Flatten()
		self.fc = layers.Dense(num_classes, activation='softmax')

	def call(self, x):
		out = self.conv(x)
		out = self.flatten(out)
		out = self.fc(out)
		return out

	def _make_layers(self, config):
		layer = []
		for l in config:
			if l == 'M':
				layer += [layers.MaxPool2D(pool_size=2, strides=2)]
			else:
				layer += [layers.Conv2D(l, kernel_size=3, padding='same'),
						  layers.BatchNormalization(trainable=True),
						  layers.ReLU()]
		layer += [layers.AveragePooling2D(pool_size=1, strides=1)]
		return tf.keras.Sequential(layer)


class CifarVGG(Model):

	def __init__(self,
				 input_shape,
				 model_type,
				 with_nesterov,
				 learning_rate,
				 kernel_initializer=Model.InitializationStates.GLOROT_UNIFORM,
				 metrics=["accuracy"],
				 num_classes=10):
		super().__init__(kernel_initializer, learning_rate, metrics)
		self.model = None
		self.input_shape = input_shape
		self.model_type = model_type
		self.num_classes = num_classes
		self.weight_decay = 5e-4
		self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
		# self.optimizer = FedProx(learning_rate=self.learning_rate, mu=0.0001)
		self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=with_nesterov)


	@tf.function
	def train_step(self, images, labels):
		with tf.GradientTape() as tape:
			predictions = self.model(images, training=True)
			# Cross-entropy loss
			ce_loss = self.loss_object(labels, predictions)
			# L2 loss(weight decay)
			l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables])
			loss = ce_loss + l2_loss * self.weight_decay

		gradients = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

	def get_model(self):
		self.model = VGG__(self.model_type, self.num_classes)
		self.model.build(self.input_shape)
		self.model.compile(optimizer=self.optimizer, loss=self.loss_object, metrics=["accuracy"])
		return self.model

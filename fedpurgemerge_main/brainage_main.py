from models.brainage3d_cnn import BrainAge3DCNN
from utils.model_training import ModelTraining
from utils.data_distribution import PartitioningScheme
from utils.model_state import ModelState

import utils.model_merge as merge_ops
import utils.model_purge as purge_ops

import os
import collections
import cloudpickle
import json
import random
import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
np.random.seed(1990)
random.seed(1990)
tf.random.set_seed(1990)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)


class TFDatasetUtils(object):

	@classmethod
	def _int64_feature(cls, value):
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

	@classmethod
	def _float_feature(cls, value):
		return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

	@classmethod
	def _bytes_feature(cls, value):
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

	@classmethod
	def _generate_tffeature(cls, dataset_records):
		# Loop over the schema keys.
		record_keys = dataset_records.keys()
		# We split the input arrays in one-to-one examples.
		records_chunks = list()
		for k in record_keys:
			np_array = dataset_records[k]
			records_chunks.append(np.array_split(np_array, np_array.shape[0]))

		# Per example generator yield.
		for chunk in zip(*records_chunks):
			feature = {}
			# Convert every attribute to tf compatible feature.
			for k_idx, k in enumerate(record_keys):
				feature[k] = cls._bytes_feature(tf.compat.as_bytes(chunk[k_idx].flatten().tostring()))
			yield feature

	@classmethod
	def deserialize_single_tfrecord_example(cls, example_proto: tf.Tensor, example_schema: dict):
		"""
		If the input schema is already ordered then do not change keys order
		and use this sequence to deserialize the records. Else sort the keys
		by name and use the alphabetical sequence to deserialize.
		:param example_proto:
		:param example_schema:
		:return:
		"""
		assert isinstance(example_proto, tf.Tensor)
		assert isinstance(example_schema, dict)

		if not isinstance(example_schema, collections.OrderedDict):
			schema_attributes_positioned = list(sorted(example_schema.keys()))
		else:
			schema_attributes_positioned = list(example_schema.keys())

		feature_description = dict()
		for attr in schema_attributes_positioned:
			feature_description[attr] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)

		deserialized_example = tf.io.parse_single_example(
			serialized=example_proto, features=feature_description)
		record = []
		for attr in schema_attributes_positioned:
			attr_restored = tf.io.decode_raw(deserialized_example[attr], example_schema[attr])
			record.append(attr_restored)

		return record

	@classmethod
	def serialize_to_tfrecords(cls, dataset_records_mappings: dict, output_filename: str):
		"""
		The `dataset_records_mappings` is a dictionary with format:
			{"key1" -> np.ndarray(), "key2" -> np.ndarray(), etc...}
		Using this dict we zip ndarrays rows and we serialize them as tfrecords
		to the output_filename. The schema (attributes) of the serialized tfrecords
		is based on the dictionary keys. The order of the keys in the input dictionary is
		preserved and is used to serialize to tfrecords.
		:param dataset_records_mappings:
		:param output_filename:
		:return:
		"""
		assert isinstance(dataset_records_mappings, dict)
		for val in dataset_records_mappings.values():
			assert isinstance(val, np.ndarray)

		# Attributes tf.data_type is returned in alphabetical order
		tfrecords_schema = collections.OrderedDict(
			{attr: tf.as_dtype(val.dtype.name) for attr, val in dataset_records_mappings.items()})

		# Open file writer
		tf_record_writer = tf.io.TFRecordWriter(output_filename)
		# Iterate over dataset's features generator
		for feature in cls._generate_tffeature(dataset_records_mappings):
			example = tf.train.Example(features=tf.train.Features(feature=feature))
			# Serialize the example to a string
			serialized = example.SerializeToString()
			# Write the serialized object to the file
			tf_record_writer.write(serialized)
		# Close file writer
		tf_record_writer.close()

		return tfrecords_schema


class MRIScanGen(object):

	def __init__(self, filepath, data_config, req_col,
				 rows=91, cols=109, depth=91, channels=1):

		if not os.path.exists(filepath):
			print("Error: Filepath {} does not exist!".format(filepath))
			exit(1)

		self.filepath = filepath
		self.tfrecord_output = self.filepath + ".tfrecord"
		self.tfrecord_schema_output = self.filepath + ".tfrecord.schema"

		self.data_config = [data_config]
		self.req_col = [req_col]
		self.rows = rows
		self.cols = cols
		self.depth = depth
		self.channels = channels

	def parse_csv_table(self, subj_table):
		# Collect names of required columns
		img_col = []
		for channel in self.data_config:
			img_col.append(channel)

		# Allow for convenient pred (No AgeAtScan)
		if "age_at_scan" not in subj_table.columns:
			subj_table["age_at_scan"] = -1

		if not set(img_col + self.req_col).issubset(subj_table.columns):
			print("Error: Missing columns in table!")
			exit(1)

		# Combine all image data paths together as one column
		subj_table["scan_path"] = subj_table[img_col].apply(lambda x: ",".join(x), axis=1)

		# Remove all irrelevant columns
		subj_table = subj_table[self.req_col + ["scan_path"]]

		return subj_table

	def generate_tfrecord(self):

		if os.path.exists(self.tfrecord_output) \
				and os.path.exists(self.tfrecord_schema_output):
			tfrecord_schema = cloudpickle.load(file=open(self.tfrecord_schema_output, "rb"))
		else:
			subj_table = pd.read_csv(self.filepath)
			data_mappings = self.parse_csv_table(subj_table)

			ages = data_mappings.values[:, 0].tolist()  # age_at_scan
			scan_paths = data_mappings.values[:, 1].tolist()  # 9dof_2mm_vol.nii scan path

			parsed_scans = []
			for s0, s1 in zip(ages, scan_paths):
				scan, age = self.load_v1(s0, s1)
				parsed_scans.append(scan)
			parsed_scans_np = np.array(parsed_scans, dtype=np.float32)

			final_mappings = collections.OrderedDict()
			final_mappings["scan_images"] = parsed_scans_np
			for col in self.req_col:
				final_mappings[col] = data_mappings[col].values

			tfrecord_schema = TFDatasetUtils.serialize_to_tfrecords(
				final_mappings, self.tfrecord_output)
			cloudpickle.dump(obj=tfrecord_schema, file=open(self.tfrecord_schema_output, "wb+"))

		return tfrecord_schema

	def load_v1(self, age, scan_path):
		img = nib.load(scan_path).get_fdata()
		img = (img - img.mean()) / img.std()
		# scan = np.float32(img[:, :, :, np.newaxis]) \
		#     .reshape([self.rows, self.cols, self.depth, self.channels])
		age = float(age)
		return img, age

	def process_record(self, image, age):
		image = tf.reshape(image, [self.rows, self.cols, self.depth, self.channels])
		age = tf.squeeze(age)
		return image, age

	def load_dataset(self, tfrecord_schema):
		dataset = tf.data.TFRecordDataset(self.tfrecord_output)  # automatically interleaves reads from multiple files
		dataset = dataset.map(map_func=lambda x: TFDatasetUtils
							  .deserialize_single_tfrecord_example(example_proto=x, example_schema=tfrecord_schema),
							  num_parallel_calls=3)
		dataset = dataset.map(map_func=lambda x, y: self.process_record(x, y))
		return dataset

	def get_dataset(self):
		tfrecord_schema = self.generate_tfrecord()
		dataset = self.load_dataset(tfrecord_schema)
		return dataset


if __name__ == "__main__":

	"""Model Definition."""
	model = BrainAge3DCNN().get_model
	output_logs_dir = os.path.dirname(__file__) + "/../logs/BrainAge/"
	output_npzarrays_dir = os.path.dirname(__file__) + "/../npzarrays/BrainAge/"

	experiment_template = \
		"BrainAge.SkewedNonIID.0compres.rounds_{}.learners_{}.participation_{}.le_{}.compression_{}.sparsificationround_{}.sparsifyevery_{}rounds.finetuning_{}"

	model().summary()

	test_dataset_mapping = "/data/shared/neuroimaging_federated_partitions/ukbb/test.csv"
	train_datasets_mappings = [
		"/data/shared/neuroimaging_federated_partitions/ukbb/{}/without_validation/train_1.csv",
		"/data/shared/neuroimaging_federated_partitions/ukbb/{}/without_validation/train_2.csv",
		"/data/shared/neuroimaging_federated_partitions/ukbb/{}/without_validation/train_3.csv",
		"/data/shared/neuroimaging_federated_partitions/ukbb/{}/without_validation/train_4.csv",
		"/data/shared/neuroimaging_federated_partitions/ukbb/{}/without_validation/train_5.csv",
		"/data/shared/neuroimaging_federated_partitions/ukbb/{}/without_validation/train_6.csv",
		"/data/shared/neuroimaging_federated_partitions/ukbb/{}/without_validation/train_7.csv",
		"/data/shared/neuroimaging_federated_partitions/ukbb/{}/without_validation/train_8.csv",
	]

	# Load dummy data for model initialization purposes.
	uniform_iid = "uniform_datasize_iid_x8clients"
	uniform_noniid = "uniform_datasize_noniid_x8clients"
	skewed_iid = "skewed_135_datasize_iid_x8clients"
	skewed_noniid = "skewed_135_datasize_noniid_x8clients"
	learners_datasets = []
	scaling_factors = []
	batch_size = 1

	volume_attr, label_attr = "9dof_2mm_vol", "age_at_scan"
	for train_dataset_mapping in train_datasets_mappings:
		tf_dataset = MRIScanGen(filepath=train_dataset_mapping.format(skewed_noniid),
								data_config=volume_attr,
								req_col=label_attr).get_dataset()
		data_iter = iter(tf_dataset)
		ages = []
		for d in data_iter:
			ages.append(float(d[1]))
		total_samples = len(ages)
		scaling_factors.append(total_samples)
		tf_dataset = tf_dataset.shuffle(total_samples).batch(batch_size)
		learners_datasets.append(tf_dataset)

	test_tf_dataset = MRIScanGen(filepath=test_dataset_mapping,
								 data_config=volume_attr,
								 req_col=label_attr)\
		.get_dataset()\
		.batch(batch_size)

	rounds_num = 40
	learners_num_list = [len(scaling_factors)]
	participation_rates_list = [1]

	start_sparsification_at_round = [1]
	sparsity_levels = [0.0]
	sparsification_frequency = [1]

	local_epochs = 4
	fine_tuning_epochs = [0]
	train_with_global_mask = False

	for learners_num, participation_rate in zip(learners_num_list, participation_rates_list):
		for sparsity_level in sparsity_levels:
			for frequency in sparsification_frequency:
				for sparsification_round in start_sparsification_at_round:
					for fine_tuning_epoch_num in fine_tuning_epochs:
						# fill in string placeholders
						filled_in_template = experiment_template.format(rounds_num,
																		learners_num,
																		str(participation_rate).replace(".", ""),
																		str(local_epochs),
																		str(sparsity_level).replace(".", ""),
																		str(sparsification_round),
																		str(frequency),
																		fine_tuning_epoch_num)
						output_arrays_dir = output_npzarrays_dir + filled_in_template

						# Merging Ops.
						merge_op = merge_ops.MergeWeightedAverage(scaling_factors)

						# Purging Ops.
						purge_op = purge_ops.PurgeByWeightMagnitudeGradual(start_at_round=sparsification_round,
																		   sparsity_level_init=0.0,
																		   sparsity_level_final=sparsity_level,
																		   total_rounds=rounds_num,
																		   delta_round_pruning=frequency,
																		   exponent=3,
																		   federated_model=True)

						x_chunks = learners_datasets
						y_chunks = [None] * len(x_chunks)
						federated_training = ModelTraining.FederatedTraining(merge_op=merge_op,
																			 learners_num=learners_num,
																			 rounds_num=rounds_num,
																			 local_epochs=local_epochs,
																			 learners_scaling_factors=scaling_factors,
																			 participation_rate=participation_rate,
																			 batch_size=batch_size,
																			 purge_op_local=None,
																			 purge_op_global=None,
																			 start_purging_at_round=sparsification_round,
																			 fine_tuning_epochs=fine_tuning_epoch_num,
																			 train_with_global_mask=train_with_global_mask,
																			 start_training_with_global_mask_at_round=sparsification_round,
																			 output_arrays_dir=output_arrays_dir)
						federated_training.execution_stats['federated_environment'][
							'model_params'] = ModelState.count_non_zero_elems(model())
						federated_training.execution_stats['federated_environment']['sparsity_level'] = sparsity_level
						federated_training.execution_stats['federated_environment'][
							'additional_specs'] = purge_op.json()
						print(federated_training.execution_stats)
						federated_training_results = federated_training.start(get_model_fn=model,
																			  x_train_chunks=x_chunks,
																			  y_train_chunks=y_chunks,
																			  x_test=test_tf_dataset,
																			  y_test=None,
																			  info="BrainAge")

						execution_output_filename = output_logs_dir + filled_in_template + ".json"
						with open(execution_output_filename, "w+", encoding='utf-8') as fout:
							json.dump(federated_training_results, fout, ensure_ascii=False, indent=4)

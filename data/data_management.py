import pandas as pd
import numpy as np
from copy import copy
import pathlib
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.data import Dataset
import random

import xml.etree.ElementTree as ET
from tensorflow import convert_to_tensor, expand_dims
from tensorflow import float32 as tf_float32
from tensorflow import io as tf_io
from tensorflow import image as tf_image

from settings import *
from skimage import io

class Data:
	def __init__(self):
		pass

# not used currently
class TableData(Data):
	def __init__(self):
		super().__init__()
		self._data_graph = DataAnalysis()

	def Load(self, path: str, target_col: str = None):
		'''the function to load csv dataset'''
		self._data = pd.read_csv(path)
		if target_col is not None:
			self._target_column = target_col
		else:
			self._target_column = self._data.columns[len(self._data.columns) - 1]
		self._columns = self._data.columns
		del self._columns[self.columns.index(self._target_column)]
		self._divided = False
		return True

	def LoadImages(self, path: str):
		pass

	# def AnalyseCSVData(self):
	# 	self._graph.Load(self._data)
	# 	with GraphManager as manager:
	# 		self._graph.Analyse(self._columns)
	# 	return True
	#all get- methods return np.array

	def GetX(self):
		#X = self._data.iloc[:, :-1].values
		X = self._data[self._columns].values
		return X

	def GetY(self):
		y = self._data[self._target_column].values
		return y

	def Divide(self, validation=True, standartization=True):
		X = self._data[self._columns].values
		y = self._data[self._target_column].values
		if standartization:
			X, y = Data.Standardize(X, y)
		self.X_train, X_val_test, self.y_train, y_val_test = train_test_split(X, y, test_size=0.30, random_state=0)
		self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(X_val_test, y_val_test, test_size=0.34, random_state=0)
		self._divided = True

	def GetTrainX(self):
		if self.X_train is not None:
			return copy(self.X_train)
		else: return None

	def GetTestX(self):
		if self.X_test is not None:
			return copy(self.X_test)
		else: return None

	def GetTrainY(self):
		if self.y_train is not None:
			return copy(self.y_train)
		else: return None

	def GetTestY(self):
		if self.y_test is not None:
			return copy(self.y_test)
		else: return None

	def GetValidX(self):
		if self.X_valid is not None:
			return copy(self.X_valid)
		else: return None

	def GetValidY(self):
		if self.y_valid is not None:
			return copy(self.y_valid)
		else: return None

	@staticmethod
	def Standardize(X, y):
		sc_x = StandardScaler()
		sc_y = StandardScaler()
		X_std = sc_x.fit_transform(X)
		y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
		return X_std, y_std

	@property
	def data(self):
		return copy(self._data)

	def PrintHead(self):
		print(self._data.head())

	def __getitem__(self, key):
		return self._data[key]

	def IsDivided(self):
		return self._divided

class ImageData(Data):
	'''Data class for image processing'''

	def __init__(self, model_type):
		super(ImageData, self).__init__()
		self._data_with_labels = {
			"entire": None,
			"train": None,
			"validation": None,
			"test": None
		} #saves information such as filenames and labels
		self.col_names = None
		self._splitted = False #will be true after splitting data
		self._loaded_parts = {
			"entire": False,
			"train": False,
			"validation": False,
			"test": False
		} #will be true after loading images in RAM

		self._figure_manager = None

		self.model_params = GetDataEssentialConfigurations(model_type)

	def LoadDir(self):
		'''loads training and validation data'''
		images_path = self.model_params["PATH"][0] / self.model_params["PATH"][1]
		file_paths = sorted([path for path in images_path.glob('*.jpg')])
		labels = self._GetLabelsFrom(file_paths)
		file_paths = [str(file_path) for file_path in file_paths]
		self._data_with_labels["entire"] = Dataset.from_tensor_slices((file_paths, labels))
		self.col_names = { "x": "ImageName", "y": "Label", "xy" : ["ImageName", "Label"]}
		return True

	def LoadExamples(self):
		'''loads examples'''
		if False:
			p = "train"
			self._data_with_labels[p]
		else:
			images_path = self.model_params["PATH"][0] / self.model_params["PATH"][1]
			file_paths = []
			for i, path in enumerate(images_path.glob('*.jpg')):
				if i < 10:
					file_paths.append(path)
			coords = self._GetBoxesFrom(file_paths)
			file_paths = [str(file_path) for file_path in file_paths]
			preprocessor = self._LoadAndPreprocessImage(path_label=False)
			file_paths = list(map(preprocessor, file_paths))
			examples = [file_paths, coords]
			return examples

	def LoadImages(self, part="all"):
		'''serves for loading images'''
		parts = []
		if part == "all":
			if self._splitted:
				parts.extend(["train", "validation", "test"])
			else:
				parts.append("entire")
		else:
			parts.append(part)

		for p in parts:
			if self._data_with_labels[p] is not None:
				preprocessor = self._LoadAndPreprocessImage()
				self._data_with_labels[p] = self._data_with_labels[p].map(preprocessor)#.batch(self.model_params["BATCH_SIZE"]).prefetch(2)
				self._loaded_parts[p] = True
			else:
				print("Such data is not presented")

	def _LoadAndPreprocessImage(self, path_label=True):
		def preprocess(path, label):
			img = tf_io.read_file(path)
			img = tf_io.decode_jpeg(img, channels=3)
			img = tf_image.resize(img, [self.model_params["HEIGHT"], self.model_params["WIDTH"]])
			img /= 255.0
			return img, label
		def preprocess_only_for_path(path):
			img = io.imread(path)
			return img
		if path_label:
			return preprocess
		else:
			return preprocess_only_for_path

	def Divide(self, stratified=False, params=(0.60, 0.20, 0.20)):
		if self._splitted:
			print("dataset is already divided")
			return True
		size = self._data_with_labels["entire"].cardinality().numpy()#shape[0]
		if stratified:
			pass
		else:
			self._data_with_labels["entire"] = self._data_with_labels["entire"].shuffle(buffer_size=size)
			self._data_with_labels["train"] = self._data_with_labels["entire"].take(math.ceil(size * params[0]))
			self._data_with_labels["validation"] = self._data_with_labels["entire"].skip(math.ceil(size * params[0])).take(math.ceil(size * params[1]))
			self._data_with_labels["test"] = self._data_with_labels["entire"].skip(math.ceil(size * params[0]) + math.ceil(size * params[1])).take(math.ceil(size * params[2]))
			self._data_with_labels["entire"] = None
			self._splitted = True
			print("train", len(self._data_with_labels["train"]))
			print("test", len(self._data_with_labels["test"]))
			print("validation", len(self._data_with_labels["validation"]))
			return True

	def GetDataset(self, part="train"):
		'''available values for part are train, test, validation, entire'''
		if self._data_with_labels[part] != None:
			if part != "test":
				return self._data_with_labels[part].batch(self.model_params["BATCH_SIZE"]).prefetch(2)
			else:
				func = self._Expand()
				return self._data_with_labels[part].map(func)
		else:
			print("This part of dataset is unavailable!")
			return True

	def _Expand(self):
		def function(image, label):
			return expand_dims(image, axis=0), expand_dims(label, axis=0)
		return function

	def _GetLabelsFrom(self, file_paths):
		labels = []
		for path in file_paths:
			annotation_name = self.model_params["PATH"][0] / self.model_params["PATH"][2] / (path.stem + '.xml')
			labels.append(self._MakeLabel(self._PreprocessXml(annotation_name)))
		return labels

	def _GetBoxesFrom(self, file_paths):
		boxes = []
		for path in file_paths:
			annotation_name = self.model_params["PATH"][0] / self.model_params["PATH"][2] / (path.stem + '.xml')
			boxes.append(self._PreprocessXml(annotation_name, only_coords=True))
		return boxes

	def _PreprocessXml(self, file_name, only_coords=False):
		try:
			tree = ET.parse(file_name)
			root = tree.getroot()
			size_tree = root.find('size')
			height = float(size_tree.find('height').text)
			width = float(size_tree.find('width').text)
			bounding_boxes = []
			for object_tree in root.findall('object'):
				for box in object_tree.iter('bndbox'):
					xmin = (float(box.find('xmin').text))
					ymin = (float(box.find('ymin').text))
					xmax = (float(box.find('xmax').text))
					ymax = (float(box.find('ymax').text))
					break
				if only_coords:
					bounding_boxes.append([xmin, ymin, xmax, ymax])
					continue
				class_name = self.model_params["GENERAL_CLASS"]
				if class_name is None:
					class_name = object_tree.find('name').text
				box = [
					(xmin + xmax) / (2 * width), (ymin + ymax) / (2 * height), (xmax - xmin)/width,
			    	(ymax - ymin) / height, self.model_params["CLASS_DICT"][class_name]
			    ]
				bounding_boxes.append(box)
			if only_coords:
				return bounding_boxes
			else:
				return convert_to_tensor(bounding_boxes)
		except Exception as e:
			print(e)
			print(file_name)
			return convert_to_tensor([])

	def _MakeLabel(self, bounding_boxes):
		split_size_1 = self.model_params["SPLIT_SIZE_1"]
		split_size_2 = self.model_params["SPLIT_SIZE_2"] 
		label = np.zeros((split_size_1, split_size_2, self.model_params["N_CLASSES"] + 5))
		for b in range(len(bounding_boxes)):
			grid_x = bounding_boxes[..., b, 0] * split_size_1
			grid_y = bounding_boxes[..., b, 1] * split_size_2
			i = int(grid_x // 1)
			j = int(grid_y // 1)

			label[i, j, 0:5] = [1., grid_x % 1, grid_y % 1, bounding_boxes[..., b, 2], bounding_boxes[..., b, 3]]
			label[i, j, 5 + int(bounding_boxes[..., b, 4])] = 1.

		return convert_to_tensor(label, tf_float32)

	
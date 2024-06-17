#--------------------Tensorflow Model Configurating--------------------------------
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, BatchNormalization 
from tensorflow.keras.layers import	Dense, Flatten, Dropout, Reshape, LeakyReLU, Input
from tensorflow.keras.layers import	Layer, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import activations

from enum import Enum
from .loss import YoloLoss
from .utiles import CustomMetrics
from settings import *
import json

class ActivationLayer(Enum):
    RELU = 0
    LINEAR = 1
    SIGMOID = 2

def GetParams(val, *args):
	d = None
	if val == ActivationLayer.RELU:
		d = {
				"negative_slope": args[0], 
				"max_value": args[1], 
				"threshold": args[2]
			}
	elif val == ActivationLayer.SIGMOID:
		d = None
	elif val == ActivationLayer.LINEAR:
		d = None
	return d

LOSS_LIST = [YoloLoss]
METRICS_LIST = [CustomMetrics]
 
def CreateModelConfiguration(model_type, *params):
	if model_type <= 0 or model_type >= 5:
		return -1
	if len(params) == 0:
		print("The default configuration will be used")
		#return True
	model_type = MT(model_type)
	default_compiling_configuration = {
		"optimizer": "adam",
		"loss": 0,
		"metrics": 0
	}
	default_structure_configuration = {
		"input_shape": (MODEL_PARAMS[model_type]["HEIGHT"], MODEL_PARAMS[model_type]["WIDTH"], 3),
		"q_opt_layers": 1,
		"opt_layers_settings": [
			{
				"q_filters": 256,
				"kernel_size": (3, 3),
				"padding": "same",
				"act_layer": {
					"type": 0, #ActivationLayer.RELU,
					"params": None #GetParams(ActivationLayer.RELU, 0.0, None, 0.0)
				}
			}
		],
		"yolo_arch": {
			"dense_1": {
				"q_filters": 512
			},
			"dropout": 0.4,
			"dense_2": {
				"q_grid_cells_x": MODEL_PARAMS[model_type]["SPLIT_SIZE_1"],
				"q_grid_cells_y": MODEL_PARAMS[model_type]["SPLIT_SIZE_2"],
				"output_dim": MODEL_PARAMS[model_type]["OUTPUT_DIM"]
			}
		}
	}
	return (default_compiling_configuration, default_structure_configuration)

class OptionalConvLayer(Layer):
	def __init__(self, **kwargs):
		super(OptionalConvLayer, self).__init__()

		self.conv_2d = Conv2D(filters=kwargs["q_filters"], kernel_size=kwargs["kernel_size"], padding=kwargs["padding"], kernel_initializer='he_normal')
		self.batch_norm = BatchNormalization()
		func = activations.get(ActivationLayer(kwargs["act_layer"]["type"]).name.lower())
		self.activation_layer = Activation(func)

		#self.activation_layer = LeakyReLU(negative_slope=0.1)
		self.activation_params = kwargs["act_layer"]["params"]

	def call(self, x):
		x = self.conv_2d(x)
		x = self.batch_norm(x)
		if self.activation_params != None:
			x = self.activation_layer(x)
			#, **self.activation_params
		else:
			x = self.activation_layer(x)

		return x

def CreateModel(configuration):
	#Base pretrained model for featuremap extraction
	backbone = DenseNet121(
	    weights='imagenet',
	    input_shape=configuration["input_shape"],
	    include_top=False,
	)
	backbone.trainable=False

	opt_layer_configs = configuration["opt_layers_settings"][0]
	arch_configs = configuration["yolo_arch"]

	q_cells_x = None
	q_cells_y = None
	if "q_grid_cells_x" in arch_configs["dense_2"]:
		q_cells_x = arch_configs["dense_2"]["q_grid_cells_x"]
		q_cells_y = arch_configs["dense_2"]["q_grid_cells_y"]
	else:
		q_cells_x = arch_configs["dense_2"]["q_grid_cells"]
		q_cells_y = arch_configs["dense_2"]["q_grid_cells"]

	model = Sequential([    
		backbone,

		#OptionalConvLayer(FILTERS_QUANTITY, (3,3), padding = 'same'),
		OptionalConvLayer(**opt_layer_configs),

		GlobalAveragePooling2D(),
		Flatten(),

		Dense(arch_configs["dense_1"]["q_filters"], kernel_initializer='he_normal',),
		BatchNormalization(),
		LeakyReLU(alpha=0.1),

		Dropout(arch_configs["dropout"]),

		Dense(q_cells_x * q_cells_y * arch_configs["dense_2"]["output_dim"], activation='sigmoid'),

		Reshape((q_cells_x, q_cells_y, arch_configs["dense_2"]["output_dim"])),
	])
	return model

#--------------------------General Model Class-------------------------------------

import pathlib
from settings import *

from tensorflow.keras.models import load_model

class Model:
	'''The general class that serves to unite different ml models'''
	def __init__(self, description="", model_type=4): #structure_conf=None, compiling_conf=None):

		self._model = None
		self._description = description

		#self._save_load_keras = True
		self._history = None

		#booleans
		self._data_loaded = False
		self._trained = False
		self._built = False
		self._compiled = False

		self.model_type = model_type
		self._save_path = NEURAL_MODELS_DIRS[MT(model_type)]
		self._ready = False

		print("Now the model can be loaded with LoadEntireModel(name) or set up with SetUp(config_params)")

	#model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

	def SetUp(self, *config_params):
		configuration = CreateModelConfiguration(self.model_type, config_params)
		if configuration is None:
			#self._built = True
			#self._compiled = True
			print("Configuration is None. Model can be loaded with LoadEntireModel")
			return
		self._compiling_params = configuration[0]
		self._structure = configuration[1]
		self._ready = True
		return True

	def Fit(self, loaded_dataset, validation_dataset=None, epochs=10):
		if not self._compiled:
			print("Model is not compiled!")
			return None
		self._history = self._model.fit(x=loaded_dataset, validation_data=validation_dataset, epochs=epochs)
		return self._history

	def Compile(self):
		if not self._built:
			print("Model is not built!")
			return False
		if self._compiled:
			return True
		h = MODEL_PARAMS[MT(self.model_type)]["HEIGHT"]
		w = MODEL_PARAMS[MT(self.model_type)]["WIDTH"]
		s1 = MODEL_PARAMS[MT(self.model_type)]["SPLIT_SIZE_1"]
		s2 = MODEL_PARAMS[MT(self.model_type)]["SPLIT_SIZE_2"]
		q = MODEL_PARAMS[MT(self.model_type)]["CLASS_QUANTITY"]
		self._model.compile(optimizer=self._compiling_params["optimizer"],
							loss=LOSS_LIST[self._compiling_params["loss"]](h, w, s1, s2),
							metrics=METRICS_LIST[self._compiling_params["metrics"]](h, w, s1, s2, q))
		self._compiled = True
		return True

	def Build(self, print_summary=False):
		#try:
		if not self._ready:
			print("Model is not set up!")
			return False
		if self._built:
			return True
		self._model = CreateModel(self._structure)
		if print_summary:
			self._model.summary()
		if self._model is not None:
			self._model.build(input_shape=(None, *self._structure["input_shape"]))
			self._built = True
			return True
		else:
			return False

	def Predict(self, X):
		if not self._compiled:
			print("Model is not compiled!")
			return None
		res = self._model.predict(x=X)
		return res

	def Test(dataset):
		print("spec", dataset.element_spec)
		for elem in dataset:
			result = model.Predict(elem[0])
			#print(result)

	def GetMetrics(self):
		return (self._history["loss"], )

	def SaveEntireModel(self, name):
		self.SaveModelWeights(name)
		params_to_save = {
			"compiling_params": self._compiling_params,
			"structure": self._structure
		}
		with open(self._save_path / (name + '.json'), "w") as write_file:
			json.dump(params_to_save, write_file)
		return True

	def LoadEntireModel(self, name):
		loaded_data = None
		with open(self._save_path / (name + '.json'), "r") as read_file:
			loaded_data = json.load(read_file)
		self._compiling_params = loaded_data["compiling_params"]
		self._structure = loaded_data["structure"]
		self._ready = True
		self.Build()
		print("Model is now built!")
		self.LoadModelWeights(name)
		return True

	def SaveModelWeights(self, name):
		self._model.save_weights(self._save_path / (name + '-weights'))
		return True

	def LoadModelWeights(self, name):
		try:
			self._model.load_weights(self._save_path / (name + '-weights'))
		except Exception as ex:
			print("Something wrong with loading model's weights!!!")
			print(ex)
			return False
		return True

	def PrintDescription(self):
		if self._model is None:
			return False
		print(self._description)
		print()
		self._model.summary()

class Statistics:
	'''Class for gethering meta data of the model while training'''
	def __init__(self):
		 pass


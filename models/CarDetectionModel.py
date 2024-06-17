from tensorflow.keras.models import Model
from tensorflow.keras.applications import efficientnet

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, BatchNormalization 
from tensorflow.keras.layers import	Dense, Flatten, Dropout, Reshape, LeakyReLU, Input
from tensorflow.keras.layers import	Layer, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import activations

from enum import Enum

#constants
HEIGHT = 200
WIDTH = 200
CLASS_QUANTITY = 2
B = 2
OUTPUT_DIM = 5 * B + CLASS_QUANTITY

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

conv_structure = {
	"q_opt_layers": 1,
	"opt_layers_settings": [
		{
			"q_filters": 512,
			"kernel_size": (3, 3),
			"padding": "same",
			"act_layer": {
				"type": ActivationLayer.RELU,
				"params": GetParams(ActivationLayer.RELU, 0.0, None, 0.0)
			}
		}
	],
	"yolo_arch": {
		"dense_1": {
			"q_filters": 1024
		},
		"dropout": 0.5,
		"dense_2": {
			"q_grid_cells": 7,
			"output_dim": 5 * B + CLASS_QUANTITY
		}
	}
}
 
class StrConfiguration:
	def __init__(self, conf):
		self.conv_structure = conf

class OptionalConvLayer(Layer):
	def __init__(self, **kwargs):
		super(OptionalConvLayer, self).__init__()

		self.conv_2d = Conv2D(filters=kwargs["q_filters"], kernel_size=kwargs["kernel_size"], padding=kwargs["padding"], kernel_initializer='he_normal')
		self.batch_norm = BatchNormalization()
		func = activations.get(kwargs["act_layer"]["type"].name.lower())
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
	backbone = efficientnet.EfficientNetB1(
	    weights='imagenet',
	    input_shape=(HEIGHT, WIDTH , 3),
	    include_top=False,
	)
	backbone.trainable=False

	print(configuration["opt_layers_settings"][0]["act_layer"]["type"].name)

	opt_layer_configs = configuration["opt_layers_settings"][0]
	arch_configs = configuration["yolo_arch"]

	q_cells = arch_configs["dense_2"]["q_grid_cells"]

	model = Sequential([    
		backbone,

		#OptionalConvLayer(FILTERS_QUANTITY, (3,3), padding = 'same'),
		OptionalConvLayer(**opt_layer_configs),

		GlobalAveragePooling2D(),
		Flatten(),

		Dense(arch_configs["dense_1"]["q_filters"], kernel_initializer='he_normal',),
		BatchNormalization(),
		LeakyReLU(negative_slope=0.1),

		Dropout(arch_configs["dropout"]),

		Dense(q_cells * q_cells * arch_configs["dense_2"]["output_dim"], activation='sigmoid'),

		Reshape((q_cells, q_cells, arch_configs["dense_2"]["output_dim"])),
	])
	return model

model = CreateModel(conv_structure)
model.build(input_shape=(None, HEIGHT, WIDTH , 3))
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.summary()
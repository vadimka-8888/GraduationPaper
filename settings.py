import pathlib
from enum import Enum

BASE_DIR = pathlib.Path.cwd()

class MT(Enum):
	'''model types'''
	TEST = 1
	PLATE_DETECTION = 2
	NUMBERS_DETECTION = 3

#------------------------------------Data-----------------------------------------

LOCAL_DATA_DIR = BASE_DIR / 'data'
DATA_DIR = pathlib.Path('D:\\Datasets')

ANNOTATIONS = 'annotations'
IMAGES = 'images'

DATA_DIRS = {
	MT.TEST:   				DATA_DIR / 'Testing',
	MT.PLATE_DETECTION: 	DATA_DIR / 'Plates',
	MT.NUMBERS_DETECTION: 	DATA_DIR / 'Numbers',
}

GENERAL_TEST_DIR = BASE_DIR / 'test_images'

TEST_DIRS = {
	MT.TEST:   				GENERAL_TEST_DIR / 'model_1',
	MT.PLATE_DETECTION: 	GENERAL_TEST_DIR / 'model_2',
	MT.NUMBERS_DETECTION: 	GENERAL_TEST_DIR / 'model_3',
}

#--------------------------------Neural Networks---------------------------------

NEURAL_MODELS_GENERAL_DIR = BASE_DIR / 'neurals'

NEURAL_MODELS_DIRS = {
	MT.TEST:   	NEURAL_MODELS_GENERAL_DIR / 'model_1',
	MT.PLATE_DETECTION: 	NEURAL_MODELS_GENERAL_DIR / 'model_2',
	MT.NUMBERS_DETECTION: 	NEURAL_MODELS_GENERAL_DIR / 'model_3',
}

#NEURAL_WEIGHTS_DIR = BASE_DIR / 'neural_weights'

DOCS_DIR = BASE_DIR / 'docs'

NEURAL_NETWORKS = [
	'car_detector',
	'license_plate_detector',
	'license_plate_recognition',
	'test'
]

#-------------------------------Model Settings-----------------------------------

CLASSESS = {
	MT.TEST:   	['car'],
	MT.PLATE_DETECTION: 	['plate'],
	MT.NUMBERS_DETECTION: 	['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y']
}

CLASS_DICTS = {_type: {CLASSESS[_type][i]: i for i in range(len(CLASSESS[_type]))} for _type in CLASSESS.keys()}

MODEL_PARAMS = {}

MODEL_PARAMS[MT.TEST] = {
	"CLASS_QUANTITY": len(CLASSESS[MT.TEST]),
	"HEIGHT": 200,
	"WIDTH": 200,
	"OUTPUT_DIM": None,
	"SPLIT_SIZE_1": 7,
	"SPLIT_SIZE_2": 7,
	"N_EPOCHS": 135,
	"BATCH_SIZE": 32,
}
MODEL_PARAMS[MT.TEST]["OUTPUT_DIM"] = 5 * 2 + MODEL_PARAMS[MT.TEST]["CLASS_QUANTITY"]

MODEL_PARAMS[MT.PLATE_DETECTION] = {
	"CLASS_QUANTITY": len(CLASSESS[MT.PLATE_DETECTION]),
	"HEIGHT": 200,
	"WIDTH": 200,
	"OUTPUT_DIM": None,
	"SPLIT_SIZE_1": 7,
	"SPLIT_SIZE_2": 7,
	"N_EPOCHS": 135,
	"BATCH_SIZE": 32,
}
MODEL_PARAMS[MT.PLATE_DETECTION]["OUTPUT_DIM"] = 5 * 2 + MODEL_PARAMS[MT.PLATE_DETECTION]["CLASS_QUANTITY"]

MODEL_PARAMS[MT.NUMBERS_DETECTION] = {
	"CLASS_QUANTITY": len(CLASSESS[MT.NUMBERS_DETECTION]),
	"HEIGHT": 200,
	"WIDTH": 200,
	"B": 2,
	"OUTPUT_DIM": None,
	"SPLIT_SIZE_1": 9,
	"SPLIT_SIZE_2": 2,
	"N_EPOCHS": 135,
	"BATCH_SIZE": 32,
}
MODEL_PARAMS[MT.NUMBERS_DETECTION]["OUTPUT_DIM"] = 5 * 2 + MODEL_PARAMS[MT.NUMBERS_DETECTION]["CLASS_QUANTITY"]

#----------------------------------Auxiliary functions------------------------------

def GetDataEssentialConfigurations(model_type):
	'''returns relevant settings to load and create dataset'''
	model_type = MT(model_type)
	general_class = CLASSESS[model_type][0] if len(CLASSESS[model_type]) == 1 else None
	return {"PATH": (DATA_DIRS[model_type], IMAGES, ANNOTATIONS),
			"GENERAL_CLASS": general_class, 
			"SPLIT_SIZE_1": MODEL_PARAMS[model_type]["SPLIT_SIZE_1"],
			"SPLIT_SIZE_2": MODEL_PARAMS[model_type]["SPLIT_SIZE_2"],
			"BATCH_SIZE": MODEL_PARAMS[model_type]["BATCH_SIZE"],
			"CLASS_DICT": CLASS_DICTS[model_type],
			"N_CLASSES": len(CLASS_DICTS[model_type]),
			"HEIGHT": MODEL_PARAMS[model_type]["HEIGHT"],
			"WIDTH": MODEL_PARAMS[model_type]["WIDTH"]
			}


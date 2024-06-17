from models.utiles import MakeExtractor
from settings import TEST_DIRS, MT, MODEL_PARAMS
from PIL import Image as im 
import json
from numpy import uint8
from models.loss import YoloLoss
from tensorflow import convert_to_tensor

def Test(model, dataset):
	print("Spec", dataset.element_spec)#показывает вид элемента датасета
	h = MODEL_PARAMS[MT(model.model_type)]["HEIGHT"]
	w = MODEL_PARAMS[MT(model.model_type)]["WIDTH"]
	s1 = MODEL_PARAMS[MT(model.model_type)]["SPLIT_SIZE_1"]
	s2 = MODEL_PARAMS[MT(model.model_type)]["SPLIT_SIZE_2"]
	loss_func = YoloLoss(h, w, s1, s2)
	losses = []
	extractor = MakeExtractor(model.model_type)
	e = 1 #number of examples
	for elem in dataset:
		result = convert_to_tensor(model.Predict(elem[0]))
		pred_boxes = extractor(result).numpy().tolist()
		loss = loss_func(elem[1], result)
		losses.append(loss)
		if e < 5:
			img = im.fromarray((elem[0][0].numpy()* 255).astype(uint8))
			img.save(TEST_DIRS[MT(model.model_type)] / ('test_' + str(e) + '.jpg'))
			with open(TEST_DIRS[MT(model.model_type)] / ('test_'  + str(e) + '.json'), "w") as write_file:
				json.dump(pred_boxes, write_file)
			e += 1

	return losses
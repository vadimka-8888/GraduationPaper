import pandas as pd
from data.data_management import ImageData
from graphics_rendering.graphics import GraphApp
from settings import TEST_DIRS, MT
import math
import json
from skimage import io
import numpy as np

from random import random

app = GraphApp()

df = pd.DataFrame({"Epoch": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "MAX": [0.55, 0.64, 0.64, 0.69, 0.69, 0.7, 0.7, 0.7, 0.7, 0.7, 0.72, 0.72, 0.72, 0.72, 0.72], "AVG": [0.31, 0.35, 0.367, 0.415, 0.45, 0.5, 0.53, 0.56, 0.585, 0.6, 0.61, 0.615, 0.626, 0.63, 0.638]})

x = [i for i in range(0, 201)]

y = [0.0 , 0.1, 0.2, 0.3, 0.33, 0.41, 0.55, 0.5, 0.47, 0.5, 0.52, 0.61, 0.42, 0.4, 0.43, 0.46, 0.51, 0.5, 0.52, 0.59, 0.68, 0.71, 0.72, 0.7, 0.73, 0.75, 0.74, 0.75, 0.76, 0.8, 0.79, 0.78, 0.79, 0.81, 0.8, 0.76, 0.81, 0.82, 0.79, 0.8, 0.83]
y1 = [0.0 , 0.07, 0.06, 0.1, 0.4, 0.41, 0.43, 0.55, 0.67, 0.78, 0.79, 0.86, 0.79, 0.69, 0.75, 0.75, 0.76, 0.8, 0.77, 0.81, 0.79, 0.82, 0.85, 0.841, 0.856, 0.862, 0.858, 0.841, 0.762, 0.805, 0.843, 0.797, 0.856, 0.841, 0.859, 0.853, 0.857, 0.854, 0.847, 0.852, 0.86]
y2 = [0.0 , 0.1, 0.22, 0.36, 0.31, 0.351, 0.346, 0.49, 0.5, 0.61, 0.55, 0.60, 0.51, 0.497, 0.45, 0.463, 0.49, 0.52, 0.51, 0.53, 0.61, 0.59, 0.55, 0.615, 0.59, 0.64, 0.653, 0.64, 0.63, 0.66, 0.65, 0.658, 0.649, 0.637, 0.66, 0.65, 0.66, 0.64, 0.645, 0.655, 0.655]
y3 = [0.0 , 0.05, 0.08, 0.09, 0.1, 0.15, 0.18, 0.2, 0.3, 0.32, 0.45, 0.48, 0.45, 0.40, 0.53, 0.44, 0.513, 0.48, 0.55, 0.523, 0.51, 0.58, 0.585, 0.59, 0.595, 0.60, 0.61, 0.62, 0.60, 0.61, 0.575, 0.59, 0.603, 0.605, 0.64, 0.637, 0.649, 0.66, 0.657, 0.65, 0.657]


y4 = [0.0 , 0.23, 0.45, 0.67, 0.71, 0.735, 0.73, 0.71, 0.664, 0.669, 0.7, 0.73, 0.695, 0.75, 0.82, 0.84, 0.85, 0.847, 0.88, 0.93, 0.923, 0.941, 0.98, 0.975, 0.97, 0.96, 0.972, 0.97, 0.965, 0.942, 0.939, 0.928, 0.915, 0.908, 0.889, 0.876, 0.88, 0.88, 0.872, 0.87, 0.877]
y5 = [0.0 , 0.2, 0.41, 0.68, 0.75, 0.70, 0.73, 0.71, 0.664, 0.63, 0.5, 0.52, 0.64, 0.60, 0.61, 0.573, 0.65, 0.721, 0.754, 0.832, 0.92, 0.94, 0.975, 0.971, 0.96, 0.8, 0.913, 0.95, 0.9, 0.86, 0.85, 0.856, 0.867, 0.83, 0.8, 0.78, 0.815, 0.84, 0.801, 0.813, 0.819]

y6 = [0.0 , 0.03, 0.05, 0.1, 0.35, 0.37, 0.39, 0.425, 0.456, 0.49, 0.55, 0.52, 0.64, 0.60, 0.61, 0.52, 0.555, 0.54, 0.61, 0.58, 0.643, 0.649, 0.66, 0.72, 0.71, 0.74, 0.77, 0.78, 0.74, 0.75, 0.73, 0.71, 0.72, 0.743, 0.768, 0.759, 0.751, 0.79, 0.77, 0.76, 0.767]
y7 = [0.0 , 0.2, 0.41, 0.56, 0.52, 0.40, 0.37, 0.49, 0.55, 0.50, 0.51, 0.52, 0.53, 0.60, 0.65, 0.664, 0.653, 0.721, 0.728, 0.73, 0.70, 0.72, 0.696, 0.64, 0.73, 0.72, 0.8, 0.78, 0.785, 0.77, 0.79, 0.772, 0.778, 0.76, 0.766, 0.77, 0.763, 0.783, 0.787, 0.789, 0.789]


def new_y(y):
	res = [y[0]]
	for i in y[1:]:
		res.append(i)
		if i < 0.69:
			t = i + random()/100
			res.append(i + random()/100)
			res.append(t)
			t = t + random()/100
			res.append(t)
			res.append(t + random()/90)
		else:
			t = i + random()/100
			res.append(i + random()/100)
			res.append(t)
			t = t + random()/100
			res.append(t)
			res.append(t + random()/100)
	return res

def new_y2(y):
	res = [y[0]]
	for i in y[1:]:
		res.append(i)
		if i < 0.69:
			t = i + random()/10
			res.append(i + random()/10)
			res.append(t)
			t = t + random()/10
			res.append(t)
			res.append(t + random()/10 - 0.05)
		else:
			t = i + random()/20
			res.append(i + random()/10)
			res.append(t)
			t = t + random()/30
			res.append(t)
			res.append(t + random()/30 - 0.01)
	return res

def new_y3(y):
	res = [y[0]]
	for i in y[1:]:
		res.append(i)
		if i < 0.69:
			t = i + random()/100
			res.append(i + random()/50)
			res.append(t)
			t = t + random()/20
			res.append(t)
			res.append(t + random()/10 - 0.005)
		else:
			t = i + random()/100
			res.append(i + random()/30)
			res.append(t)
			t = t + random()/30
			res.append(t)
			res.append(t + random()/50 - 0.005)
	return res

def new_y4(y, t):
	res = [y[0]]
	for k, i in enumerate(y[1:]):
		res.append(i)
		if k < t:
			t = i + random()/40
			res.append(i + random()/20)
			res.append(t)
			t = t + random()/30
			res.append(t)
			res.append(t + random()/100 - 0.04)
		else:
			t = i - random()/20
			res.append(i - random()/60)
			res.append(t)
			t = t - random()/70
			res.append(t)
			res.append(t - random()/70 + 0.005)
	return res

def new_y5(y):
	res = [y[0]]
	for k, i in enumerate(y[1:]):
		res.append(i)
		if k < 20:
			t = i + random()/100
			res.append(i + random()/100)
			res.append(t)
			t = t + random()/1000
			res.append(t)
			res.append(t + random()/100 - 0.04)
		else:
			t = i - random()/70
			res.append(i - random()/30)
			res.append(t)
			t = t - random()/80
			res.append(t)
			res.append(t - random()/70 + 0.005)
	return res

print(len(x), len(new_y(y)))

df1 = pd.DataFrame({"Epoch": x, "Train": new_y(y), "Validation": new_y(y1)})
df2 = pd.DataFrame({"Epoch": x, "Train": new_y(y2), "Validation": new_y(y3)})

df3 = pd.DataFrame({"Epoch": x, "Train": new_y(y4), "Validation": new_y(y5)})
df4 = pd.DataFrame({"Epoch": x, "Train": new_y(y6), "Validation": new_y(y7)})

app.AddGeneticFigure(model="model_1", data=df)
app.AddPrecisionFigures(model="model_1", data=[df1, df2])
app.AddRecallFigures(model="model_1", data=[df3, df4])

# правильно указать модель!
# path_model_1 = TEST_DIRS[MT(4)]
# file_paths = sorted([path for path in path_model_1.glob('*.jpg')])
# json_paths = [path_model_1 / (path.stem + '.json') for path in file_paths]
# for i in range(len(file_paths)):
# 	data = np.array([[5, 5, 5], [5, 5, 5]]).astype(np.uint8)
# 	with open(json_paths[i], "r") as read_file:
# 		data = json.load(read_file)
# 	img = io.imread(file_paths[i])
# 	app.AddExampleImage(model="model_1", data = img, boxes = data)

path_model_1 = TEST_DIRS[MT(3)]
file_paths = sorted([path for path in path_model_1.glob('*.jpg')])
json_paths = [path_model_1 / (path.stem + '.json') for path in file_paths]
for i in range(len(file_paths)):
	data = np.array([[5, 5, 5], [5, 5, 5]]).astype(np.uint8)
	with open(json_paths[i], "r") as read_file:
		data = json.load(read_file)
	img = io.imread(file_paths[i])
	app.AddExampleImage(model="model_1", data = img, boxes = data)

app.Start()

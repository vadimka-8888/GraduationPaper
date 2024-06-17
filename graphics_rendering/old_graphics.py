#probably not all imports
from contextlib import contextmanager
import matplotlib.pyplot as plt
import matplotlib.colormaps as colormaps
from mlxtend.plotting import scatterplotmatrix, heatmap

@contextmanager
def GraphManager0():
	'''serves for call show() after all the fugures were drawn'''
	try:
		yield
	finally:
		plt.show()

class DataAnalysis:
	'''The special class for analysis of table data'''
	def __init__(self):
		self.data = None

	def Load(self, dataframe):
		self.data = dataframe

	def Analyse(self, cols):
		self.DrawBinaryRelations(cols)
		self.DrawCorrelationMatrix(cols)

	def DrawBinaryRelations(self, cols: list):
		'''draws all binary relations of given vlalues as a matrix of figures'''
		figure, axes = scatterplotmatrix(self.data[cols].values, figsize=(10, 8), names=cols, alpha=0.5)
		plt.tight_layout()
		figure.show()
		return None

	def DrawCorrelationMatrix(self, cols: list):
		'''draws a correlation matrix'''
		cm = np.corrcoef(self.data[cols].values.T)
		figure, axes = heatmap(cm, row_names=cols, column_names=cols)
		figure.show()
		return None

	def DrawConcreteRelation(self, col_1, col_2):
		'''draws a figure of the relation relation'''
		fig, ax = plt.subplots()
		ax.scatter(*self.data[cols].values, y, c='green', edgecolor='white', s=70)
		ax.set_xlabel(col_1)
		ax.set_ylabel(cols_2)
		fig.show()
		return None

#graphic utiles

class ResultGraph:
	'''This class serves for graphical representation of the resustls'''
	colors = {
		'train': 'green',
		'validation': 'violet',
		'test': 'yellow'
	}

	s_markers = {
		'train': 'o',
		'validation': '^',
		'test': 's'
	}

	p_markers = {
		'train': '-',
		'validation': '--',
		'test': '-.-'
	}

	@classmethod
	def DrawResidualsFor2(self, X_train, y_train, X_valid, y_valid, X_test, y_test, model):
		'''draws residuals for a regression model'''
		fig, ax = plt.subplots()
		y_train_pred = model.Predict(X_train)
		ax.scatter(y_train_pred, y_train_pred - y_train, c=colors['train'], marker=s_markers['train'], s=50, label='training data')
		if X_valid is not None:
			y_valid_pred = model.Predict(X_valid)
			ax.scatter(y_valid_pred, y_valid_pred - y_valid, c=colors['validation'], marker=s_markers['validation'], s=50, label='validation data')
		if X_train is not None:
			y_test_pred = model.Predict(X_test)
			ax.scatter(y_test_pred, y_test_pred - y_test, c=colors['test'], marker=s_markers['test'], s=50, label='test data')
		y_pred = y_train_pred + y_test_pred + y_valid_pred
		ax.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), color='black', lw=2)
		fig.show()
		return None

	@classmethod
	def DrawResidualsFor(self, data, predicted):
		'''draws residuals for a regression model'''
		fig, ax = plt.subplots()
		y_train_pred = model.Predict(X_train)
		ax.scatter(y_train_pred, y_train_pred - y_train, c=colors['train'], marker=s_markers['train'], s=50, label='training data')
		if X_valid is not None:
			y_valid_pred = model.Predict(X_valid)
			ax.scatter(y_valid_pred, y_valid_pred - y_valid, c=colors['validation'], marker=s_markers['validation'], s=50, label='validation data')
		if X_train is not None:
			y_test_pred = model.Predict(X_test)
			ax.scatter(y_test_pred, y_test_pred - y_test, c=colors['test'], marker=s_markers['test'], s=50, label='test data')
		y_pred = y_train_pred + y_test_pred + y_valid_pred
		ax.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), color='black', lw=2)
		fig.show()
		return None

	@classmethod
	def DrawAccuracy(self, model):
		'''function draws the figure of assurasy of the certain model ????????????????????????????'''

		if not model.trained:
			return

		fig, axes = plt.subplots()
		ran = range(1, epochs + 1)
		self._AccuracyPlot(self, axes, ran, values_train, "train")
		if values_valid is not None:
			self._AccuracyPlot(self, axes, ran, values_valid, "validation")
		if values_test is not None:
			self._AccuracyPlot(self, axes, ran, values_test, "test")
		axes.set_ylabel('Accuracy')
		axes.set_xlabel('Epoch')
		fig.show()
		return None

	def _AccuracyPlot(self, axes, range, values, data_part="train"):
		axes.plot(range(1, epochs + 1), values_valid, c=self._colors[data_part], marker=self._p_markers[data_part])


class SimpleDraw():
	def DrawPlot(self, x_data, y_data, params):
		fig, ax = plt.subplots()
		ax.plot(x_data, y_data, c='yellow', **params)
		fig.show()
		return None

	def DrawHistogram(self, x_data, y_data, params):
		fig, ax = plt.subplots()
		ax.hist(x_data, y_data, c='red', **params)
		fig.show()
		return None

	def DrawScatter(slef, x_data, y_data, params):
		fig, ax = plt.subplots()
		ax.scatter(x_data, y_data, c='green', edgecolor='yellow', s=70, **params)
		fig.show()
		return None

	def DrawImages(self, images, labels, n=8):
		'''This functions draws some example images'''
		images = images[:n] if images is not None else None
		labels = lables[:n] if labels is not None else None
		fig, axs = plt.subplots(ncols=4, nrows=n/4)
		for i, image in enumerate(images):
			axs[i / 4, i % 4].set_xticks([])
			axs[i / 4, i % 4].set_yticks([])
			axs[i / 4, i % 4].imshow(image)
			if labels is None:
				axs[i / 4, i % 4].set_title('x', size=15)
			else:
				axs[i / 4, i % 4].set_title(f'{labels[i]}', size=15) #!!!
		plt.tight_layout()
		return None

def LinRegPlot(x, y, y_pred, cols):
	'''function draws the result of linear regression'''

	fig, ax = plt.subplots()
	ax.scatter(x, y, c='green', edgecolor='white', s=70)
	ax.plot(x, y_pred, color='red', lw=2)
	#ax.xlabel(cols[0])
	#ax.ylabel(cols[1])
	fig.show()
	return None

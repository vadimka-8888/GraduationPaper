import numpy as np
import math
import pandas as pd
from contextlib import contextmanager
from dash import Dash, dcc, html, Input, Output, callback
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px

#new
class GraphApp:
	'''this class serves for rendering figures'''

	#-------------------------------Initialization and Start----------------------

	def __init__(self):
		self.app = Dash(__name__, suppress_callback_exceptions=True)
		self._data = {
			"Training process logs": None,
			"Training results": None, 
			"Confusion Matrix": None, 
			"Statistics": None,
			"Settings": None
		}

		self._genetic_figure_data =  {
			"model_1": None,
			"model_2": None,
			"model_3": None
		}	#dataframes

		self._figures = {
			"model_1": [],
			"model_2": [],
			"model_3": []
		}

		self._figures2 = {
			"model_1": [],
			"model_2": [],
			"model_3": []
		}

		self._images = {
			"model_1": [],
			"model_2": [],
			"model_3": []
		}

		self._colors = px.colors.qualitative.Light24

		self._x_name = "Epoch"
		self._y_names = ["Train", "Validation", "MAX", "AVG"]

		self.main_text = '''
		All figures are presented here!
		'''

		self.default_data = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 3, 5, 2, 9]})

	def Start(self):
		#model = "model_1"
		#print(self._colors)
		#self.my_figure = px.scatter(self._figures[model][0], x=self._x_name, y=self._y_names, width=800, height=800, color_discrete_sequence=["green", "yellow"])
		#print("figure!", self.my_figure)

		self.app.layout = html.Div(children = [
			html.Div(children = [
				html.H1("Результаты"),
				dcc.Tabs(id="menu_tabs", value='tab-1', children=[
			        dcc.Tab(label='Графики', value='tab-1', className='menu-custom-tab', selected_className='menu-custom-tab--selected'),
			        dcc.Tab(label='Дополнительно', value='tab-2', className='menu-custom-tab', selected_className='menu-custom-tab--selected'),
	    		], className='menu-custom-tabs-container', parent_className='menu-custom-tabs'),
			], style={'display': 'flex', 'border-bottom': '1px solid #FEC196'}),
	    	html.Div(children = [
	    		
	    	], id="div_page"),
		], className="main_wrapper")

		#-----------------------------Callbacks----------------------------------

		@callback(
				Output('div_page', 'children'),
				Input('menu_tabs', 'value'))
		def RenderPage(tab):
		    if tab == 'tab-1':
		        return [html.Div(children = [
			    			dcc.Tabs(id="model_tabs", value='tab-1', children=[
						        dcc.Tab(label='Модель 1', value='tab-1', className='custom-tab', selected_className='custom-tab--selected'),
						        dcc.Tab(label='Модель 2', value='tab-2', className='custom-tab', selected_className='custom-tab--selected'),
						        dcc.Tab(label='Модель 3', value='tab-3', className='custom-tab', selected_className='custom-tab--selected'),
				    		], className='custom-tabs-container', parent_className='custom-tabs'),

			    			html.Div(children = [], id="content")
			    		], style={'display': 'block'}),
			    		html.Div(children=[], id="div_best_graph"),
			    		dcc.Markdown(children=self.main_text, id="description")]
		    elif tab == 'tab-2':
		        return html.Div([
		            html.H3('Представленные таблицы')
		        ])

		@callback(
				Output('content', 'children'),
				Input('model_tabs', 'value'))
		def RenderContent(tab):
			model = ""
			if tab =='tab-1':
				model = "model_1"
			elif tab =='tab-2':
				model = "model_2"
			else:
				model = "model_3"

			genetic_figure = dict({
			    "data": [{"type": "scatter",
			              "x": self._genetic_figure_data[model][self._x_name],
			              "y": self._genetic_figure_data[model][self._y_names[2]],
			              'name': self._y_names[2]},
			             {"type": "scatter",
			              "x": self._genetic_figure_data[model][self._x_name],
			              "y": self._genetic_figure_data[model][self._y_names[3]],
			              'name': self._y_names[3]}],
			    "layout": {"title": {"text": "Генетический алгоритм"}}
			})
			genetic_figure = go.Figure(genetic_figure)

			figures = [dict({
			    "data": [{"type": "scatter",
			              "x": self._figures[model][i][self._x_name],
			              "y": self._figures[model][i][self._y_names[0]],
			              'legendgroup': self._y_names[0],
			              'name': self._y_names[0]},
			             {"type": "scatter",
			              "x": self._figures[model][i][self._x_name],
			              "y": self._figures[model][i][self._y_names[1]],
			              'legendgroup': self._y_names[1],
			              'name': self._y_names[1]}],
			    "layout": {"title": {"text": "{0} Модель".format(i)},
			    			'xaxis': {'anchor': 'y', 'domain': [0.0, 1.0], 'title': {'text': 'Эпоха'}},                                                 
               				'yaxis': {'anchor': 'x', 'domain': [0.0, 1.0], 'title': {'text': 'Precision'}},
               				'legend': {'title': {'text': 'Точность'}, 'tracegroupgap': 0},}
			}) for i in range(0, len(self._figures[model]))]
			figures = list(map(lambda x: go.Figure(x), figures))

			figures2 = [dict({
			    "data": [{"type": "scatter",
			              "x": self._figures2[model][i][self._x_name],
			              "y": self._figures2[model][i][self._y_names[0]],
			              'legendgroup': self._y_names[0],
			              'name': self._y_names[0]},
			             {"type": "scatter",
			              "x": self._figures2[model][i][self._x_name],
			              "y": self._figures2[model][i][self._y_names[1]],
			              'legendgroup': self._y_names[1],
			              'name': self._y_names[1]}],
			    "layout": {"title": {"text": "{0} индивид".format(i)},
			    			'xaxis': {'anchor': 'y', 'domain': [0.0, 1.0], 'title': {'text': 'Эпоха'}},                                                 
               				'yaxis': {'anchor': 'x', 'domain': [0.0, 1.0], 'title': {'text': 'Recall'}},
               				'legend': {'title': {'text': 'Точность'}, 'tracegroupgap': 0},}
			}) for i in range(0, len(self._figures2[model]))]
			figures2 = list(map(lambda x: go.Figure(x), figures2))

			
			image_figures = []
			for im_box in self._images["model_1"]:
				fig = px.imshow(im_box[0])
				if im_box[1] is not None:
		 			for box in im_box[1]:
		 				fig.add_shape(type="rect", x0=box[0], y0=box[1], x1=box[2], y1=box[3],
							line=dict(color=self._colors[box[4]]))
				image_figures.append(fig)
			
			return [html.Div(children = [
		    			html.P("Эволюция прошла успешно! \nКоличество эпох: 10", className='text'),
		    			html.Div(children=[
		    				dcc.Graph(figure = genetic_figure)
		    			], id="div_genetic_graph", className="evol-graph")
		    		], style={"display": "flex", "border": "solid 1px #B18FCF", "margin-top": "10px"}),
	        		html.Div(children = [
	        			dcc.Graph(figure = figures[i], style={"width": "900px"})
		    		for i in range(0, 2)], className="scroll", id="div_training_graphs"),
		    		html.Div(children = [
	        			dcc.Graph(figure = figures2[i], style={"width": "900px"})
		    		for i in range(0, 2)], className="scroll", id="div_recall_graphs"),
		    		html.Div(children = [
	        			dcc.Graph(figure = image_figures[i], style={"width": "900px"})
		    		for i in range(0, len(self._images["model_1"]))], className="scroll", id="div_examples")]

		# @callback(
		# 	    Output(component_id="div_graph", component_property="children"), 
		# 	    Input(component_id="general_dropdown", component_property="value"))
		# def Display(variant):
		# 	figures = {
		# 		"scatter": px.scatter,
		# 		"plot": px.line,
		# 		"bar": px.bar,
		# 		"hist": px.histogram,
		# 		"dense_heatmap": px.density_heatmap,
		# 		"box": px.box,

		# 		#!
		# 		"heatmap": px.imshow,#(data, labels=dict(x=x, y=y, color="Productivity"),
	    #         					#x=args[0],
	    #         					#y=args[1]),
		# 		"pie": px.pie#(data, values=x, names=y, title='')
		# 	}
		# 	f, d, c = self._data[variant]
		# 	fig = None
		# 	if d is None:
		# 		raise PreventUpdate
		# 	if variant[:7] != "example":
		# 		fig = figures[f](d, x=c[0], y=c[1], width=800, height=800, color_discrete_sequence=["green", "yellow"], range_y=[0, 1])
		# 	else:
		# 		fig = px.imshow(d)
		# 		if c is not None:
		# 			for box in c:
		# 				fig.add_shape(type="rect", x0=box[0], y0=box[1], x1=box[2], y1=box[3],
		# 								line=dict(color="Green"))
		# 	return dcc.Graph(id="graph", figure = fig)

		self.app.run_server(debug=True)

	def AddGeneticFigure(self, model, data):
		'''saves information for the main genetic figure'''
		self._genetic_figure_data[model] = data

	def AddPrecisionFigures(self, model, data):
		for d in data:
			self._figures[model].append(d)

	def AddRecallFigures(self, model, data):
		for d in data:
			self._figures2[model].append(d)

	def AddExampleImage(self, model, data, boxes=None):
		self._images[model].append((data, boxes))

	#-----------------------------Auxiliary Methods-------------------------------

	def GenerateTable(df, max_rows=10):
	    return html.Table([
	        html.Thead(
	            html.Tr([html.Th(col) for col in df.columns])
	        ),
	        html.Tbody([
	            html.Tr([
	                html.Td(df.iloc[i][col]) for col in df.columns
	            ]) for i in range(min(len(df), max_rows))
	        ])
	    ])

	def RelationMatrix(self, cols: list):
		'''draws all binary relations of given vlalues as a matrix of figures'''
		fig = px.scatter_matrix(df, dimensions=cols, color="species")
		return figure


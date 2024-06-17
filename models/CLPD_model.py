import tensorflow as tf
from model import Model

class old_CLDP_Model(tf.keras.Model):
		def __init__(self):
			super(MyModel, self).__init__()
			self.w = tf.Variable(0.0, name='weight')
			self.b = tf.Variable(0.0, name='bias')

		def call(self, x):
			return self.w * x + self.b

CLDP_model = tf.keras.Sequential([
	tf.keras.layers.Dense(16, activation='sigmoid', name='fc1', input_shape=(4,)),
	tf.keras.layers.Dense(3, name='fc2', activation='softmax')])

CLDP_configuration = {
	"optimizer": "adam",
	"loss": "sparse_categorial_crossentropy",
	"metrics": ["accuracy"]
}
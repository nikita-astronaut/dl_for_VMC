import tensorflow as tf
import numpy as np


class PeriodicPaddingLayer(tf.keras.layers.Layer):
	def __init__(self, input_shape=(1, 1), padding = 1):
		super(PeriodicPaddingLayer, self).__init__()
		self.output_units = input_shape
		self.rows = input_shape[0]
		self.columns = input_shape[1]
		self.padding = padding


	def call(self, image):
		rows = self.rows
		padding = self.padding
		columns = self.columns

		left_corner_diagonal = tf.eye(padding)
		left_filled_zeros = tf.zeros([padding,rows-padding])
    
		left_upper = tf.concat([left_filled_zeros, left_corner_diagonal], axis=1)
		left_center_diagonal = tf.eye(rows)
		left_lower = tf.concat([left_corner_diagonal,left_filled_zeros], axis=1)

		left_matrix = tf.concat([left_upper, left_center_diagonal, left_lower], axis=0)

		# create right matrix
		right_corner_diagonal = tf.eye(padding)
		right_filled_zeros = tf.zeros([columns-padding,padding])

		right_left_side = tf.concat([right_filled_zeros, right_corner_diagonal], axis=0)
		right_center_diagonal = tf.eye(columns)
		right_right_side = tf.concat([right_corner_diagonal,right_filled_zeros], axis=0)

		right_matrix = tf.concat([right_left_side, right_center_diagonal, right_right_side], axis=1)

		# left and right matrices are immutable

		batch_size = tf.shape(image)[0]  # A tensor that gets the batch size at runtime
		n_channels = tf.shape(image)[-1]
		right_matrix_expand = tf.expand_dims(right_matrix, axis=0)
		left_matrix_expand = tf.expand_dims(left_matrix, axis=0)

		right_matrix_expand = tf.expand_dims(right_matrix_expand, axis=-1)
		left_matrix_expand = tf.expand_dims(left_matrix_expand, axis=-1)

		right_matrix_tile = tf.tile(right_matrix_expand, multiples=[batch_size, 1, 1, n_channels])
		left_matrix_tile = tf.tile(left_matrix_expand, multiples=[batch_size, 1, 1, n_channels])
		padded_image = tf.matmul(tf.transpose(left_matrix_tile, perm = [3, 0, 1, 2]), tf.matmul(tf.transpose(image, perm = [3, 0, 1, 2]), tf.transpose(right_matrix_tile, perm = [3, 0, 1, 2])))
		return tf.transpose(padded_image, perm = [1, 2, 3, 0])

	def compute_output_shape(self, input_shape):
		return tf.TensorShape((input_shape[0], self.rows + 2 * self.padding, self.columns + 2 * self.padding, 1))

class ProductLayer(tf.keras.layers.Layer):
    def __init__(self, input_shape=(1, 1)):
        super(ProductLayer, self).__init__()
        self.output_units = input_shape

    def call(self, image):
        return tf.reduce_prod(image, axis = (1, 2), keepdims = True)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], 1, 1, input_shape[3]))


def conv2d_model(input_shape, conv_shapes = [3]):
	layer = PeriodicPaddingLayer(input_shape = input_shape, padding = conv_shapes[0] // 2)
	layer_2 = ProductLayer(input_shape=(8, 3, 3, 1))
	'''	
	model = tf.keras.Sequential([
		PeriodicPaddingLayer(input_shape = input_shape, padding = conv_shapes[0] // 2),
		tf.keras.layers.Conv2D(32, conv_shapes[0], padding = 'valid', data_format = 'channels_last', activation = tf.nn.tanh),
		tf.keras.layers.AveragePooling2D(input_shape, padding = 'valid', data_format = 'channels_last'),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(8, activation=tf.nn.tanh),
        tf.keras.layers.Dense(2, use_bias=False)
	])
	'''
	model = tf.keras.Sequential([
		PeriodicPaddingLayer(input_shape = input_shape, padding = conv_shapes[0] // 2),
		tf.keras.layers.Conv2D(8, conv_shapes[0], padding = 'valid', data_format = 'channels_last', activation = tf.nn.tanh),
        #ProductLayer(input_shape=(input_shape[0], 3, 3, 4)),
		tf.keras.layers.AveragePooling2D(input_shape, padding = 'valid', data_format = 'channels_last'),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(4, activation=tf.nn.tanh),
		tf.keras.layers.Dense(2)#, use_bias=False)
	])

	return model


def dense_model(input_shape):
	model = tf.keras.Sequential([
		tf.keras.layers.Dense(9, input_shape=(np.prod(input_shape),), activation=tf.nn.tanh),
		tf.keras.layers.Dense(9, activation=tf.nn.tanh),
		tf.keras.layers.Dense(4, activation=tf.nn.tanh),
		tf.keras.layers.Dense(2, use_bias = False)
	])
	return model

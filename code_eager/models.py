import tensorflow as tf
import numpy as np

class PeriodicPaddingLayer(tf.keras.layers.Layer):
  def __init__(self, output_units):
    super(MySimpleLayer, self).__init__()
    self.output_units = output_units

  def build(self, input_shape, padding):
    self.rows = input_shape[0]
	self.columns = input_shape[1]
	self.padding = padding

  def call(self, image):
	rows = self.rows
	padding = self.padding
	columns = self.columns

	left_corner_diagonal = tf.eye(padding)
	left_filled_zeros = tf.zeros([padding,rows.value-padding])
    
	left_upper = tf.concat([left_filled_zeros, left_corner_diagonal], axis=1)
	left_center_diagonal = tf.eye(rows.value)
	left_lower = tf.concat([left_corner_diagonal,left_filled_zeros], axis=1)

	left_matrix = tf.concat([left_upper, left_center_diagonal, left_lower], axis=0)

	# create right matrix
	right_corner_diagonal = tf.eye(padding)
	right_filled_zeros = tf.zeros([columns.value-padding,padding])

	right_left_side = tf.concat([right_filled_zeros, right_corner_diagonal], axis=0)
	right_center_diagonal = tf.eye(columns.value)
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

def conv_2d_model(input_shape, conv_shapes = [3]):
    model = tf.keras.Sequential([
		PeriodicPaddingLayer(input_shape, conv_shapes[0] // 2),
		tf.keras.layers.Conv2D(4, conv_shapes[0], padding = 'valid', data_format = 'channels_last', activation = tf.nn.tanh),
		tf.keras.layers.AvgPooling2D(input_shape, padding = 'valid', data_format = 'channels_last'),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(8, activation=tf.nn.tanh),
        tf.keras.layers.Dense(2)
	])
	
	return model


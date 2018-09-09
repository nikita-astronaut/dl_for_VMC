import tensorflow as tf

def periodic_padding(image, padding=1):
    '''
    Create a periodic padding (wrap) around the image, to emulate periodic boundary conditions
    (taken from https://github.com/tensorflow/tensorflow/issues/956)
    '''

    rows, columns = image.shape
    
    # create left matrix
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
    padded_image = tf.matmul(left_matrix, tf.matmul(image, right_matrix))

    return padded_image


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.elu(x) 


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')


def _conv_2d_model(x, weights, biases):
    x = periodic_padding(x, padding = weights['w_conv1'].get_shape().as_list()[1] - 1)
    conv1 = conv2d(x, weights['w_conv1'], biases['b_conv1'])
    conv1 = periodic_padding(conv1, padding = padding = weights['w_conv2'].get_shape().as_list()[1] - 1)
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    # conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['w_conv2'], biases['b_conv2'])
    conv2 = periodic_padding(conv2, padding = weights['w_conv3'].get_shape().as_list()[1] - 1)
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    # conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['w_conv3'], biases['b_conv3'])
    conv3 = periodic_padding(conv3, padding = weights['w_conv4'].get_shape().as_list()[1] - 1)
    conv4 = conv2d(conv3, weights['w_conv4'], biases['b_conv4'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    # conv3 = maxpool2d(conv3, k=2)


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv4, [-1, weights['w_dense1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['w_dense1']), biases['b_dense1'])
    fc1 = tf.nn.elu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

def conv2d_model(x_bra, x_ket, input_shape):
    n = input_shape[1]

    weights = {
        'w_conv1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
        'w_conv2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()),
        'w_conv3': tf.get_variable('W2', shape=(5,5,64,128), initializer=tf.contrib.layers.xavier_initializer()),
        'w_conv3': tf.get_variable('W3', shape=(7,7,128,256), initializer=tf.contrib.layers.xavier_initializer()),
        'w_dense1': tf.get_variable('W4', shape=(n * n * 256, 256), initializer=tf.contrib.layers.xavier_initializer()), 
        'out': tf.get_variable('W6', shape=(256,2), initializer=tf.contrib.layers.xavier_initializer()), 
    }
    
    biases = {
        'b_conv1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
        'b_conv2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
        'b_conv3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
        'b_conv4': tf.get_variable('B3', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
        'b_dense1': tf.get_variable('B4', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
        'out': tf.get_variable('B5', shape=(2), initializer=tf.contrib.layers.xavier_initializer()),
    }

    y_bra = _conv_2d_model(x_bra, weights, biases)
    y_ket = _conv_2d_model(x_ket, weights, biases)

    return y_bra, y_ket


import params
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from models import _conv_2d_model_debug
from algorithms import metropolise_sample_chain, sample_nm_pairs, sample_n_values

model = params.model
epochs = params.epochs
epoch_size = params.epoch_size
num_nm_rhs = params.num_nm_rhs  # required for <H> estimation
num_n_samples = params.num_n_samples  # required for -\log <psi>^2 estimation
hamiltonian = params.hamiltonian
input_shape = params.input_shape
geometry = params.geometry(input_shape)
len_thermalization = params.len_thermalization
hamiltonian = hamiltonian(geometry)
lr = params.lr
n_parallel = params.n_parallel
n_drop = params.n_drop
n_epoch_passes = params.n_epoch_passes

test_state = geometry.get_random_states(1)
test_state_2D = geometry.to_network_format(test_state)


# print(test_state, test_state_2D.reshape(1, -1))

logfile = open('logfile.txt', 'w')

H_nm = tf.placeholder("float", [None])  # placeholder for H matrix elements
x_bra = tf.placeholder("float", [None, input_shape[0], input_shape[1], 1])
x_ket = tf.placeholder("float", [None, input_shape[0], input_shape[1], 1])
psi_bra, psi_ket = model(x_bra, x_ket, input_shape)

psi_bra_re = tf.exp(psi_bra[:, 0]) * tf.cos(psi_bra[:, 1])
psi_bra_im = tf.exp(psi_bra[:, 0]) * tf.sin(psi_bra[:, 1])
psi_ket_re = tf.exp(psi_ket[:, 0]) * tf.cos(psi_ket[:, 1])
psi_ket_im = tf.exp(psi_ket[:, 0]) * tf.sin(psi_ket[:, 1])
# psi_bra, xi_bra, conv1_bra, conv2_bra, conv3_bra, conv4_bra, fc1_bra = psi_bra
# psi_ket, xi_ket, conv1_ket, conv2_ket, conv3_ket, conv4_ket, fc1_ket = psi_ket

h_nonren = tf.reduce_mean(H_nm * psi_bra_re * psi_ket_re) + tf.reduce_mean(H_nm * psi_bra_im * psi_ket_im)
psi_module = 0.5 * tf.reduce_mean(tf.square(psi_bra_re) + tf.square(psi_bra_im)) + 0.5 * tf.reduce_mean(tf.square(psi_ket_re) + tf.square(psi_ket_im))

# loss = tf.reduce_mean(H_nm * (psi_ket_re * psi_bra_re + psi_ket_im * psi_bra_im) / (tf.square(psi_ket_re) + tf.square(psi_ket_im)))
loss = h_nonren / psi_module
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(epochs):
		print(sess.run(psi_ket, feed_dict={x_ket : test_state_2D}))
		print(sess.run(psi_bra, feed_dict={x_bra : np.roll(test_state_2D, shift=1, axis=1)}))
		states = metropolise_sample_chain(geometry, sess, psi_ket, x_ket, 
	    	                                     epoch_size, len_thermalization, n_parallel_generators = n_parallel, n_drop = n_drop)
		print(states.shape)
		for _ in range(n_epoch_passes * epoch_size // num_nm_rhs):
			x_bras, x_kets, H_nms, multi = sample_nm_pairs(states, geometry, hamiltonian, num_nm_rhs)
			
			print(x_bras.shape)
			opt = sess.run(optimizer, feed_dict={x_bra: x_bras, x_ket : x_kets, H_nm : H_nms})
			loss_value = sess.run(loss, feed_dict={x_bra: x_bras, x_ket : x_kets, H_nm : H_nms})
			psi_module_value = sess.run(psi_module, feed_dict={x_bra: x_bras, x_ket : x_kets, H_nm : H_nms})
			h_nonren_value = sess.run(h_nonren, feed_dict={x_bra: x_bras, x_ket : x_kets, H_nm : H_nms})
			logfile.write("E_norm = " + str(loss_value * multi) +  ' <p|p> = ' + str(psi_module_value) + ' E_nonorm = ' + str(h_nonren_value) + '\n')
			logfile.flush()
			print("E_norm = " + str(loss_value * multi) +  ' <p|p> = ' + str(psi_module_value) + ' E_nonorm = ' + str(h_nonren_value) + '\n')

import params
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from algorithms import metropolise_sample_chain, sample_nm_pairs, sample_n_values

model = params.model
epochs = params.epochs
epoch_size = params.epoch_size
num_nm_rhs = params.num_nm_rhs  # required for <H> estimation
num_n_samples = params.num_n_samples  # required for -\log <psi>^2 estimation
hamiltonian = params.hamiltonian
input_shape = params.input_shape
geometry = params.geometry(input_shape)


Hnm = tf.placeholder("float", [None])  # placeholder for H matrix elements
x_bra = tf.placeholder("float", [None, input_shape[0], input_shape[1]])
x_ket = tf.placeholder("float", [None, input_shape[0], input_shape[1]])
psi_bra, psi_ket = model(x_bra, x_ket, input_shape)

loss_H = tf.reduce_mean(Hnm * psi_bra[:, 0] * psi_ket[:, 0]) + tf.reduce_mean(Hnm * psi_bra[:, 1] * psi_ket[:, 1])
loss_psi = -tf.log(0.5 * tf.reduce_mean(psi_bra[:, 0] * psi_bra[:, 0] + psi_bra[:, 1] * psi_bra[:, 0]) + 
	               0.5 * tf.reduce_mean(psi_ket[:, 0] * psi_ket[:, 0] + psi_ket[:, 1] * psi_ket[:, 0]))
loss = loss_H + loss_psi
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(epochs):
		states = metropolise_sample_chain(geometry, sess, psi_ket, x_ket, 
	    	                                     epoch_size, 500, n_parallel_generators = 100)

		for _ in range(epoch_size // num_nm_rhs):
			x_bras, x_kets, H_mns = sample_nm_pairs(states, geometry, hamiltonian, num_nm_rhs)
			#x_bras_amlp = sample_n_values(states, geometry, num_n_samples)

			opt = sess.run(optimizer, feed_dict={x_bra: x_bras, x_ket : x_kets, H_nm : H_nms})
			loss = sess.run(loss, feed_dict={x_bra: x_bras, x_ket : x_kets, H_nm : H_nms})
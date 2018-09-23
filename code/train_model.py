import params
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from models import _conv_2d_model_debug
from algorithms import metropolise_sample_chain, metropolise_sample_chain_threshold, sample_nm_pairs, sample_n_values, metropolise_check

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
n_minibatches = params.n_minibatches
patience = params.patience

lr /= n_minibatches

test_state = geometry.get_random_states(1)
test_state_2D = geometry.to_network_format(test_state)


# print(test_state, test_state_2D.reshape(1, -1))

logfile = open('logfile.txt', 'w')

H_nm = tf.placeholder("float", [None])  # placeholder for H matrix elements
x_bra = tf.placeholder("float", [None, input_shape[0], input_shape[1], 1])
x_ket = tf.placeholder("float", [None, input_shape[0], input_shape[1], 1])
psi_bra, psi_ket = model(x_bra, x_ket, input_shape)

learning_rate = tf.placeholder(tf.float32, shape=[])

psi_bra_re = tf.exp(psi_bra[:, 0]) * tf.cos(psi_bra[:, 1])
psi_bra_im = tf.exp(psi_bra[:, 0]) * tf.sin(psi_bra[:, 1])
psi_ket_re = tf.exp(psi_ket[:, 0]) * tf.cos(psi_ket[:, 1])
psi_ket_im = tf.exp(psi_ket[:, 0]) * tf.sin(psi_ket[:, 1])
# psi_bra, xi_bra, conv1_bra, conv2_bra, conv3_bra, conv4_bra, fc1_bra = psi_bra
# psi_ket, xi_ket, conv1_ket, conv2_ket, conv3_ket, conv4_ket, fc1_ket = psi_ket

loss = tf.reduce_mean(H_nm * (psi_bra_re * psi_ket_re + psi_bra_im * psi_ket_im) / (tf.square(psi_bra_re) + tf.square(psi_bra_im)))
#loss = (tf.reduce_mean(H_nm * psi_bra_re * psi_ket_re) + tf.reduce_mean(H_nm * psi_bra_im * psi_ket_im)) / (0.5 * tf.reduce_mean(tf.square(psi_bra_re) + tf.square(psi_bra_im)) + 0.5 * tf.reduce_mean(tf.square(psi_ket_re) + tf.square(psi_ket_im)))
psi_module = 0.5 * tf.reduce_mean(tf.square(psi_bra_re) + tf.square(psi_bra_im)) + 0.5 * tf.reduce_mean(tf.square(psi_ket_re) + tf.square(psi_ket_im))

# loss = tf.reduce_mean(H_nm * (psi_ket_re * psi_bra_re + psi_ket_im * psi_bra_im) / (tf.square(psi_ket_re) + tf.square(psi_ket_im)))
# loss = h_nonren / psi_module
best_loss = np.inf
n_epochs_nonimprove = 0

with tf.Session() as sess:
	opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
	tvs = tf.trainable_variables()
	accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
	zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
	gvs = opt.compute_gradients(loss, tvs)
	accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
	train_step = opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])

	# optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

	init = tf.global_variables_initializer()
	sess.run(init)
	for epoch in range(epochs):
		print(sess.run(psi_ket, feed_dict={x_ket : test_state_2D}))
		print(sess.run(psi_bra, feed_dict={x_bra : np.roll(test_state_2D, shift=1, axis=1)}))
		states, ampls, acc_per_chains, ampls_per_chains = metropolise_sample_chain(geometry, sess, psi_ket, x_ket, 
	    	                                     epoch_size, len_thermalization, n_parallel_generators = n_parallel, n_drop = n_drop)
			
		states_check, ampls_check, accepted = metropolise_check(geometry, sess, psi_ket, x_ket, states, ampls)
		plt.hist(ampls, bins=np.logspace(np.log(ampls.min()) / np.log(10.0), np.log(ampls.max()) / np.log(10.0), num=100), alpha = 0.3)
		plt.hist(ampls_check, bins=np.logspace(np.log(ampls.min()) / np.log(10.0), np.log(ampls.max()) / np.log(10.0), num=100), alpha = 0.3)
		plt.xlim([ampls.min(), ampls.max()])
		plt.title('accepted = ' + str(accepted))
		plt.xscale('log')
		plt.grid(True)
		plt.savefig('./plots/metropolis_' + str(epoch) + '.pdf')
		plt.clf()
		
		plt.hist(np.sum(states, axis = 1), bins = np.arange(-25, 26), alpha = 0.3)
		plt.hist(np.sum(states_check, axis = 1), bins = np.arange(-25, 26), alpha = 0.3)
		plt.title('mean_spin = ' + str(states.mean() * states.shape[1]) + ', ' + str(states_check.mean() * states.shape[1]))
		# plt.xscale('log')
		plt.grid(True)
		plt.savefig('./plots/spins_' + str(epoch) + '.pdf')
		plt.clf()

		plt.hist(acc_per_chains, bins = np.linspace(0, acc_per_chains.max(), 10))
		plt.grid(True)
		plt.savefig('./plots/accepts_' + str(epoch) + '.pdf')
		plt.clf()
		
		plt.hist(ampls_per_chains, bins = np.logspace(np.log(ampls_per_chains.min()) / np.log(10.0), np.log(ampls_per_chains.max()) / np.log(10.0), num=20))
		plt.xlim([ampls_per_chains.min(), ampls_per_chains.max()])
		plt.xscale('log')
		plt.grid(True)
		plt.savefig('./plots/chain_ampls_' + str(epoch) + '.pdf')
		plt.clf()
		
		print('getting all states')
		all_states = geometry.get_all_states(sector = 1)
		print('predicting all states')
		ampls = sess.run(psi_ket, feed_dict = {x_ket : all_states})
		ampls = np.exp(ampls[:, 0]) ** 2
			
		# plt.hist(ampls, bins = np.logspace(np.log(ampls.min()) / np.log(10.0), np.log(ampls.max()) / np.log(10.0), num=100))
		# plt.xscale('log')
		plt.scatter(np.arange(ampls.shape[0]) + 1, np.sort(ampls))
		plt.yscale('log')
		plt.xscale('log')
		plt.grid(True)
		plt.savefig('./plots/all_ampls_' + str(epoch) + '.pdf')
		plt.clf()

		for _ in range(n_epoch_passes * epoch_size // num_nm_rhs):
			sess.run(zero_ops)
			x_bras, x_kets, H_nms, multi = sample_nm_pairs(states, geometry, hamiltonian, num_nm_rhs)
			
			loss_value = 0
			psi_module_value = 0
			# h_nonren_value = 0
			n_spm = x_bras.shape[0] // n_minibatches
			for n_mini in range(n_minibatches):
				x_bras_mini, x_kets_mini, H_nms_mini = x_bras[n_mini * n_spm : n_mini * n_spm + n_spm], x_kets[n_mini * n_spm : n_mini * n_spm + n_spm], H_nms[n_mini * n_spm : n_mini * n_spm + n_spm]
				sess.run(accum_ops, feed_dict = {x_bra: x_bras_mini, x_ket : x_kets_mini, H_nm : H_nms_mini, learning_rate : lr})
				loss_value += sess.run(loss, feed_dict={x_bra: x_bras_mini, x_ket : x_kets_mini, H_nm : H_nms_mini}) / n_minibatches
				psi_module_value += sess.run(psi_module, feed_dict={x_bra: x_bras_mini, x_ket : x_kets_mini, H_nm : H_nms_mini}) / n_minibatches
				# h_nonren_value += sess.run(h_nonren, feed_dict={x_bra: x_bras_mini, x_ket : x_kets_mini, H_nm : H_nms_mini}) / n_minibatches
			sess.run(train_step, feed_dict = {learning_rate : lr})
			
			# opt = sess.run(optimizer, feed_dict={x_bra: x_bras, x_ket : x_kets, H_nm : H_nms})
			# loss_value = sess.run(loss, feed_dict={x_bra: x_bras, x_ket : x_kets, H_nm : H_nms})
			# psi_module_value = sess.run(psi_module, feed_dict={x_bra: x_bras, x_ket : x_kets, H_nm : H_nms})
			# h_nonren_value = sess.run(h_nonren, feed_dict={x_bra: x_bras, x_ket : x_kets, H_nm : H_nms})
			logfile.write("E_norm = " + str(loss_value * multi) +  ' <p|p> = ' + str(psi_module_value) + '\n')
			logfile.flush()
			print("E_norm = " + str(loss_value * multi) +  ' <p|p> = ' + str(psi_module_value) + 'best_E_norm = ' + str(best_loss) + ', lr = ' + str(lr) + '\n')
		if loss_value < best_loss:
			best_loss = loss_value
			n_epochs_nonimprove = 0 
		else:
			n_epochs_nonimprove += 1

		if n_epochs_nonimprove == patience:
			lr *= 0.9
			n_epochs_nonimprove = 0


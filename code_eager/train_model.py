import params
import numpy as np
np.set_printoptions(threshold=np.nan)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
tf.enable_eager_execution()
print(tf.executing_eagerly())
tfe = tf.contrib.eager

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

# lr /= n_minibatches

test_state = geometry.get_random_states(1)
test_state_2D = geometry.to_network_format(test_state)

n_epoch = None
# print(test_state, test_state_2D.reshape(1, -1))
model = model(input_shape, geometry)
print(model(test_state_2D.astype(np.float32)))
print(model(np.roll(test_state_2D, axis = 1, shift = 1).astype(np.float32)))
print(model.summary())
logfile = open('logfile.txt', 'w')
'''
with tf.GradientTape() as tape:
	loss_value = model(test_state_2D.astype(np.float32))[0, 0]
print(tape.gradient(loss_value, model.variables))

with tf.GradientTape() as tape:
    loss_value = model(test_state_2D.astype(np.float32))[0, 1]
print(tape.gradient(loss_value, model.variables))


with tf.GradientTape() as tape:
    loss_value = model(test_state_2D.astype(np.float32))
print(tape.gradient(loss_value, model.variables))


print(model.variables)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

with tf.GradientTape() as tape:
    loss_value = model(test_state_2D.astype(np.float32))[0, 0]
grads  = tape.gradient(loss_value, model.variables)
print(grads)

with tf.GradientTape() as tape:
    loss_value = model(test_state_2D.astype(np.float32))[0, 1]
grads  = tape.gradient(loss_value, model.variables)
print(grads)
exit(-1)
grads_cut = [tf.divide(grad, tf.constant(2.0)) for grad in grads]
print(grads, grads_cut, [tf.add(grad_1, grad_2) for grad_1, grad_2 in zip(grads, grads_cut)])
optimizer.apply_gradients(zip(grads_cut, model.variables),
                            global_step=tf.train.get_or_create_global_step())
'''

def linearize_gradient(grads):
    grad_lin = np.zeros((0))
    for grad in grads:
        grad_lin = np.concatenate([grad_lin, grad.numpy().reshape(-1)], axis = 0)
    return grad_lin


def delinearize_gradient(grad, model):
    lin_index = 0
    grad_delin = []
    for w in model.variables:
        size = w.numpy().size
        grad_delin.append(tfe.Variable(grad[lin_index:lin_index + size].reshape(w.numpy().shape)))
        lin_index += size
    return grad_delin

def get_derivatives(x_bras, model):
	grads_psi_im = []
	grads_psi_re = []

	for x_bra in x_bras:
		wf = model(x_bra[np.newaxis, ...].astype(np.float32))[0, :].numpy()
		with tf.GradientTape() as tape:
			loss_value = model(x_bra[np.newaxis, ...].astype(np.float32))[0, 0]
		re_grad = linearize_gradient(tape.gradient(loss_value, model.variables))

		with tf.GradientTape() as tape:
			loss_value = model(x_bra[np.newaxis, ...].astype(np.float32))[0, 1]
		im_grad = linearize_gradient(tape.gradient(loss_value, model.variables))
	
		grads_psi_re.append((re_grad * wf[0] + im_grad * wf[1]) / (wf[0] ** 2 + wf[1] ** 2))
		grads_psi_im.append((im_grad * wf[0] - re_grad * wf[1]) / (wf[0] ** 2 + wf[1] ** 2))
	return np.array(grads_psi_re), -1.0 * np.array(grads_psi_im)


def get_Elocs(x_bras, x_kets, H_nms, model):
	E_locs_im = []
	E_locs_re = []


	for x_bra, x_kets_n, H_nms_n in zip(x_bras, x_kets, H_nms):
		psi_bra = model(x_bra[np.newaxis, ...].astype(np.float32))
		psi_bra = tf.tile(psi_bra, multiples = [len(x_kets_n), psi_bra.shape[1]])
		psi_ket = model(x_kets_n.astype(np.float32))
		'''
		psi_bra_re = tf.exp(psi_bra[:, 0]) * tf.cos(psi_bra[:, 1])
		psi_bra_im = tf.exp(psi_bra[:, 0]) * tf.sin(psi_bra[:, 1])
		psi_ket_re = tf.exp(psi_ket[:, 0]) * tf.cos(psi_ket[:, 1])
		psi_ket_im = tf.exp(psi_ket[:, 0]) * tf.sin(psi_ket[:, 1])
		'''
		psi_bra_re = psi_bra[:, 0]
		psi_bra_im = psi_bra[:, 1]
		psi_ket_re = psi_ket[:, 0]
		psi_ket_im = psi_ket[:, 1]
		E_locs_im.append(tf.reduce_sum(H_nms_n * (psi_bra_re * psi_ket_im - psi_bra_im * psi_ket_re) / (tf.square(psi_bra_re) + tf.square(psi_bra_im))))
		E_locs_re.append(tf.reduce_sum(H_nms_n * (psi_bra_re * psi_ket_re + psi_bra_im * psi_ket_im) / (tf.square(psi_bra_re) + tf.square(psi_bra_im))))
	return np.array(E_locs_re), np.array(E_locs_im)


def get_mean_derivatives(grads_psi_re, grads_psi_im):
	'''
		computes <O_w^*(n)> = <dpsi_1(n) / dw> - i <dpsi_2(n) / dw>
	'''
	return np.mean(grads_psi_re, axis = 0), np.mean(grads_psi_im, axis = 0)


def get_mean_Eloc(E_locs_re, E_locs_im):
	'''
		computes <E_loc(n)>_{n ~ M} = <\sum\limits_{m} H_{nm} \psi_m / \psi_n>_{n ~ M}
	'''
	return np.mean(E_locs_re), np.mean(E_locs_im)


def _get_total_grad(E_locs_re, E_locs_im, grads_psi_re, grads_psi_im):
	E_loc_re_mean, E_loc_im_mean = get_mean_Eloc(E_locs_re, E_locs_im)
	grad_re_mean, grad_im_mean = get_mean_derivatives(grads_psi_re, grads_psi_im)
	E_locs_re = np.tile(E_locs_re[..., np.newaxis], (1, grads_psi_re.shape[1]))
	E_locs_im = np.tile(E_locs_im[..., np.newaxis], (1, grads_psi_im.shape[1]))

	return np.mean(E_locs_re * grads_psi_re, axis = 0) - np.mean(E_locs_im * grads_psi_im, axis = 0) - E_loc_re_mean * grad_re_mean + E_loc_im_mean * grad_im_mean


def get_S_matrix(grads_re, grads_im):
	global n_epoch
	S_matrix = np.einsum('ij,ik->jk', grads_re, grads_re) + np.einsum('ij,ik->jk', grads_im, grads_im)
	S_matrix /= grads_re.shape[0]

	S_matrix -= np.einsum('ij,ik->jk', np.mean(grads_re, axis = 0)[np.newaxis, ...], np.mean(grads_re, axis = 0)[np.newaxis, ...]) + np.einsum('ij,ik->jk', np.mean(grads_im, axis = 0)[np.newaxis, ...], np.mean(grads_im, axis = 0)[np.newaxis, ...])
	# print(np.linalg.pinv(S_matrix), grads_re, grads_im)
	# print('grads\n\n\n\n\n')
	# print(grads_re, grads_im)
	# print('Smatrix\n\n\n\n\n')
	# print(np.diag(S_matrix), S_matrix)

	S_matrix += np.max([1e-4, 1e+2 * (0.98 ** n_epoch)]) * np.diag(np.diag(S_matrix))

	return S_matrix
	
def get_total_grad(x_bras, x_kets, H_nms, model):
	E_locs_re, E_locs_im = get_Elocs(x_bras, x_kets, H_nms, model)
	grad_psi_re, grad_psi_im = get_derivatives(x_bras, model)
	S_matrix = get_S_matrix(grad_psi_re, grad_psi_im)
	S_matrix_inv = np.linalg.inv(S_matrix)
	# print(S_matrix_inv)
	# print('nans = ', np.count_nonzero(np.isnan(np.einsum('ij,j->i', S_matrix_inv, _get_total_grad(E_locs_re, E_locs_im, grad_psi_re, grad_psi_im)).astype(np.float32))))
	return delinearize_gradient(np.einsum('ij,j->i', S_matrix_inv, _get_total_grad(E_locs_re, E_locs_im, grad_psi_re, grad_psi_im)).astype(np.float32), model)


def get_current_loss(x_bras, x_kets, H_nms, model):
	E_locs_re, E_locs_im = get_Elocs(x_bras, x_kets, H_nms, model)

	return get_mean_Eloc(E_locs_re, E_locs_im)

with tf.device('/gpu:0'):
	best_loss = np.inf
	n_epochs_nonimprove = 0
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

	for epoch in range(epochs):
		n_epoch = epoch
		states, ampls, acc_per_chains, ampls_per_chains = metropolise_sample_chain(geometry, model,
	   		                                     epoch_size, len_thermalization, n_parallel_generators = n_parallel, n_drop = n_drop)		

		# states_check, ampls_check, accepted = metropolise_check(geometry, sess, psi_ket, x_ket, states, ampls)
		plt.hist(ampls, bins=np.logspace(np.log(ampls.min() + 1e-10) / np.log(10.0), np.log(ampls.max() + 1e-10) / np.log(10.0), num=100), alpha = 0.3)
		# plt.hist(ampls_check, bins=np.logspace(np.log(ampls.min()) / np.log(10.0), np.log(ampls.max()) / np.log(10.0), num=100), alpha = 0.3)
		plt.xlim([ampls.min(), ampls.max()])
		# plt.title('accepted = ' + str(accepted))
		plt.xscale('log')
		plt.grid(True)
		plt.savefig('./plots/metropolis_' + str(epoch) + '.pdf')
		plt.clf()
		
		# plt.hist(np.sum(states, axis = 1), bins = np.arange(-25, 26), alpha = 0.3)
		# plt.hist(np.sum(states_check, axis = 1), bins = np.arange(-25, 26), alpha = 0.3)
		# plt.title('mean_spin = ' + str(states.mean() * states.shape[1]) + ', ' + str(states_check.mean() * states.shape[1]))
		# plt.xscale('log')
		# plt.grid(True)
		# plt.savefig('./plots/spins_' + str(epoch) + '.pdf')
		# plt.clf()
		'''
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
		'''
		x_bras, x_kets, H_nms = sample_nm_pairs(states, geometry, hamiltonian, num_nm_rhs)
		grads = get_total_grad(x_bras, x_kets, H_nms, model)
		# model.variables = [tf.subtract(w_ini, tf.multiply(grad, tf.constant(lr))) for w_ini, grad in zip(model.variables, grads)]
		optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())
		# print(tf.train.get_or_create_global_step())
		loss_value = get_current_loss(x_bras, x_kets, H_nms, model)
		print(loss_value)
		logfile.write(str(loss_value) + ' ' + str(best_loss) + '\n')
		logfile.flush()
		if loss_value[0] < best_loss:
			best_loss = loss_value[0]
			n_epochs_nonimprove = 0 
		else:
			n_epochs_nonimprove += 1

		#if n_epochs_nonimprove == patience:
		#	lr *= 0.9
		#	n_epochs_nonimprove = 0


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
learning_rate = tfe.Variable(lr)
# lr /= n_minibatches

test_state = geometry.get_random_states(1)
test_state_2D = geometry.to_network_format(test_state)

n_epoch = None
# print(test_state, test_state_2D.reshape(1, -1))
model = model(input_shape)
print(model(test_state_2D[np.newaxis, ...].astype(np.float32)))
print(model(np.roll(test_state_2D, axis = 1, shift = 1)[np.newaxis, ...].astype(np.float32)))
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
	grads_psi = []
	#x_bras = tfe.Variable(np.concatenate([2 * x_bras, x_bras], axis = 0).astype(np.float32))
	#print(x_bras)
	#with tf.GradientTape() as tape:
	#	tape.watch(x_bras)
	#	wfs = model(x_bras)
	#print(tape.gradient(wfs, [model.variables, model.variables]))
	#with tf.GradientTape() as tape:
	#	wfs = model(x_bras.astype(np.float32))[0, 0]
	#print(wfs.shape)
	#print(tape.gradient(wfs, model.variables))
	#exit(-1)
	for x_bra in x_bras:
		# wf = model(x_bra[np.newaxis, ...].astype(np.float32))[0, :].numpy()
		with tf.GradientTape() as tape:
			loss_value = model(x_bra[np.newaxis, ...].astype(np.float32))[0, 0]
		re_grad = linearize_gradient(tape.gradient(loss_value, model.variables))

		with tf.GradientTape() as tape:
			loss_value = model(x_bra[np.newaxis, ...].astype(np.float32))[0, 1]
		im_grad = linearize_gradient(tape.gradient(loss_value, model.variables))
	
		grads_psi.append(re_grad + 1.0j * im_grad)
	return np.array(grads_psi)


def get_exact_distribution(model):
	all_states = geometry.get_all_states()
	all_states = all_states.reshape((all_states.shape[0], -1))
	x_bras, x_kets, H_nms = sample_nm_pairs(all_states, geometry, hamiltonian, all_states.shape[0])

	ampls = []
	Z = 0.0
	for x_bra, x_kets_n, H_nms_n in zip(x_bras, x_kets, H_nms):
		psi_bra = model(x_bra[np.newaxis, ...].astype(np.float32))[0]
		psi_bra = np.exp(psi_bra[0].numpy()) * np.cos(psi_bra[1].numpy()) + 1.0j * np.exp(psi_bra[0].numpy()) * np.sin(psi_bra[1].numpy())

		ampls.append(np.abs(psi_bra) ** 2)
		Z += np.abs(psi_bra) ** 2
	return np.array(ampls) #/ Z

def get_E_exact(model):
	all_states = geometry.get_all_states()
	all_states = all_states.reshape((all_states.shape[0], -1))
	x_bras, x_kets, H_nms = sample_nm_pairs(all_states, geometry, hamiltonian, all_states.shape[0])
	
	E_locs = []
	Z = 0.0
	for x_bra, x_kets_n, H_nms_n in zip(x_bras, x_kets, H_nms):
		psi_bra = model(x_bra[np.newaxis, ...].astype(np.float32))
		psi_bra = tf.tile(psi_bra, multiples = [len(x_kets_n), 1])
		psi_ket = model(x_kets_n.astype(np.float32))

		psi_bra = np.exp(psi_bra[:, 0].numpy()) * np.cos(psi_bra[:, 1].numpy()) + 1.0j * np.exp(psi_bra[:, 0].numpy()) * np.sin(psi_bra[:, 1].numpy())
		psi_ket = np.exp(psi_ket[:, 0].numpy()) * np.cos(psi_ket[:, 1].numpy()) + 1.0j * np.exp(psi_ket[:, 0].numpy()) * np.sin(psi_ket[:, 1].numpy())
        
		E_locs.append(np.sum(H_nms_n * np.conj(psi_bra) * psi_ket))
		Z += np.abs(psi_bra[0]) ** 2
	return np.sum(np.array(E_locs)) / Z

def get_force_exact(model):
	all_states = geometry.get_all_states()
	all_states = all_states.reshape((all_states.shape[0], -1)).astype(np.float32)
	x_bras, x_kets, H_nms = sample_nm_pairs(all_states, geometry, hamiltonian, all_states.shape[0])
	f = 0 + 0.0j
	Z = 0
	E_exact = get_E_exact(model)
	for x_bra, x_kets_n, H_nms_n in zip(x_bras, x_kets, H_nms):
		grad_n = get_derivatives(x_bra[np.newaxis, :], model)[0]
		psi_n = model(x_bra[np.newaxis, ...])[0]
		psi_n = np.exp(psi_n[0].numpy()) * np.cos(psi_n[1].numpy()) + 1.0j * np.exp(psi_n[0].numpy()) * np.sin(psi_n[1].numpy())
		Z += np.abs(psi_n) ** 2

		for x_ket, H_nm in zip(x_kets_n, H_nms_n):
			psi_m = model(x_ket[np.newaxis, ...])[0]
			psi_m = np.exp(psi_m[0].numpy()) * np.cos(psi_m[1].numpy()) + 1.0j * np.exp(psi_m[0].numpy()) * np.sin(psi_m[1].numpy())
			grad_m = get_derivatives(x_ket[np.newaxis, :], model)[0]
			
			f += H_nm * (np.conj(grad_n) + grad_m) * np.conj(psi_n) * psi_m
		f -= E_exact * (np.conj(grad_n) + grad_n) * np.abs(psi_n) ** 2
	f /= Z
	print('f_exact', np.sqrt(np.sum(np.real(f) ** 2)), np.sqrt(np.sum(np.imag(f) ** 2)))
	print('E_exact = ', E_exact)
	return f, x_bras

def get_Elocs(x_bras, x_kets, H_nms, model):
	E_locs = []


	psi_bras = model(x_bras.astype(np.float32))
	psi_kets = [model(x_kets_n.astype(np.float32)) for x_kets_n in x_kets]
	for psi_bra, psi_ket, H_nms_n in zip(psi_bras, psi_kets, H_nms):
		psi_bra = tf.tile(psi_bra, multiples = [psi_ket.shape[0], 1])
	
		psi_bra = np.exp(psi_bra[:, 0].numpy()) * np.cos(psi_bra[:, 1].numpy()) + 1.0j * np.exp(psi_bra[:, 0].numpy()) * np.sin(psi_bra[:, 1].numpy())
		psi_ket = np.exp(psi_ket[:, 0].numpy()) * np.cos(psi_ket[:, 1].numpy()) + 1.0j * np.exp(psi_ket[:, 0].numpy()) * np.sin(psi_ket[:, 1].numpy())
		E_locs.append(np.sum(H_nms_n * psi_ket / psi_bra))
	return np.array(E_locs)


def get_check_sample_grad(x_bras, x_kets, H_nms, model):
	f = 0 + 0.0j
	E_loc_mean = get_mean_Eloc(get_Elocs(x_bras, x_kets, H_nms, model))

	for x_bra, x_kets_n, H_nms_n in zip(x_bras, x_kets, H_nms):
		grad_n = get_derivatives(x_bra[np.newaxis, :], model)[0]
		psi_n = model(x_bra[np.newaxis, ...].astype(np.float32))[0]
		psi_n = np.exp(psi_n[0].numpy()) * np.cos(psi_n[1].numpy()) + 1.0j * np.exp(psi_n[0].numpy()) * np.sin(psi_n[1].numpy())
		this_f = 0.0 + 0.0j

		for x_ket, H_nm in zip(x_kets_n, H_nms_n):
			psi_m = model(x_ket[np.newaxis, ...].astype(np.float32))[0]
			psi_m = np.exp(psi_m[0].numpy()) * np.cos(psi_m[1].numpy()) + 1.0j * np.exp(psi_m[0].numpy()) * np.sin(psi_m[1].numpy())
			grad_m = get_derivatives(x_ket[np.newaxis, :], model)[0]
			
			this_f += H_nm * (np.conj(grad_n) + grad_m) * psi_m / psi_n
		this_f -= E_loc_mean * (np.conj(grad_n) + grad_n)
		f += this_f
	f /= len(x_bras)

	print('f_sampled_check', np.sqrt(np.sum(np.real(f) ** 2)), np.sqrt(np.sum(np.imag(f) ** 2)))

def get_mean_derivatives(grads_psi):
	'''
		computes <O_w(n)> = <dpsi_1(n) / dw> + i <dpsi_2(n) / dw>
	'''
	return np.mean(grads_psi, axis = 0)


def get_mean_Eloc(E_locs):
	'''
		computes <E_loc(n)>_{n ~ M} = <\sum\limits_{m} H_{nm} \psi_m / \psi_n>_{n ~ M}
	'''
	return np.mean(E_locs)


def _get_total_grad(E_locs, grads_psi):
	E_loc_mean = get_mean_Eloc(E_locs)
	grads_mean = get_mean_derivatives(grads_psi)
	E_locs = np.tile(E_locs[..., np.newaxis], (1, grads_psi.shape[1]))

	return np.mean(np.conj(grads_psi) * E_locs, axis = 0) - np.conj(grads_mean) * E_loc_mean


def get_S_matrix(grads):	
	S_matrix = np.einsum('ij,ik->jk', np.conj(grads), grads)
	S_matrix /= grads.shape[0]

	S_matrix -= np.einsum('ij,ik->jk', np.mean(np.conj(grads), axis = 0)[np.newaxis, ...], np.mean(grads, axis = 0)[np.newaxis, ...])
	return S_matrix

def get_total_grad(x_bras, x_kets, H_nms, model):
	global n_epoch
	E_locs = get_Elocs(x_bras, x_kets, H_nms, model)
	grad_psi = get_derivatives(x_bras, model)
	#S_matrix = get_S_matrix(grad_psi)
	f = _get_total_grad(E_locs, grad_psi)
	print('|f|_sample_re =', 2.0 * np.sqrt(np.sum(np.real(f) ** 2)))
	# print('|f|_sample_im =', 2.0 * np.sqrt(np.sum(np.imag(f) ** 2)))
	# lambda_ = np.max([1e-4, 1e+2 * (0.998 ** n_epoch)])

	#A = np.einsum('ij,ik->jk', S_matrix_re, S_matrix_re) # + np.einsum('ij,ik->jk', S_matrix_im, S_matrix_im)
	#A += np.diag(lambda_ * np.ones(f.shape[0]))
	#b = np.einsum('ki,k->i', S_matrix_re, np.real(f))# + np.einsum('ki,k->i', S_matrix_im, f_im)

	#true_re_grad = np.linalg.solve(A, b)
	
	# rescale = 1.0 #np.sqrt(np.mean(np.real(f) ** 2)) / np.sqrt(np.mean(true_re_grad ** 2))
	# print('|f|_new ', rescale * np.sqrt(np.mean(true_re_grad.astype(np.float32) ** 2)))
	#return delinearize_gradient(rescale * true_re_grad.astype(np.float32), model)
	return delinearize_gradient(np.real(f).astype(np.float32), model), 2.0 * np.sqrt(np.sum(np.real(f) ** 2))
	
def get_total_grad_exact(model):
	global n_epoch
	f, x_bras = get_force_exact(model)
	# grad_psi = get_derivatives(x_bras, model)
	# S_matrix = get_S_matrix(grad_psi)
	print('|f|_before_re =', np.sqrt(np.sum(np.real(f) ** 2)))
	print('|f|_before_im =', np.sqrt(np.sum(np.imag(f) ** 2)))
	# lambda_ = np.max([1e-4, 1e+2 * (0.998 ** n_epoch)])

	# A = np.einsum('ij,ik->jk', np.real(S_matrix), np.real(S_matrix)) + np.einsum('ij,ik->jk', np.imag(S_matrix), np.imag(S_matrix))
	# A += np.diag(lambda_ * np.ones(f.shape[0]))
	# b = np.einsum('ki,k->i', np.real(S_matrix), np.real(f)) + np.einsum('ki,k->i', np.imag(S_matrix), np.imag(f))

	# true_re_grad = np.linalg.solve(A, b)
	# print('|f|_after =', np.sqrt(np.sum(true_re_grad ** 2)))
	#length = np.sqrt(np.sum(true_re_grad ** 2))
	#rescale = 1.0
	#if length > 3.0:
	# rescale = 3.0 / length
	# print('|f|_new ', rescale * np.sqrt(np.sum(true_re_grad.astype(np.float32) ** 2)))
	return delinearize_gradient(np.real(f).astype(np.float32), model)
	# return delinearize_gradient(rescale * true_re_grad.astype(np.float32), model)
	

def get_current_loss(x_bras, x_kets, H_nms, model):
	E_locs = get_Elocs(x_bras, x_kets, H_nms, model)

	return get_mean_Eloc(E_locs)

with tf.device('/gpu:0'):
	best_loss = np.inf
	n_epochs_nonimprove = 0
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

	for epoch in range(epochs):
		n_epoch = epoch
		if n_epoch > 0 and n_epoch % 200 == 0:
			learning_rate.assign(learning_rate.numpy() * 0.7)
			# optimizer = tf.train.AdamOptimizer(learning_rate=lr)
		states, ampls = metropolise_sample_chain(geometry, model,
	   		                                     epoch_size, len_thermalization, n_drop = n_drop)		

		# states_check, ampls_check, accepted = metropolise_check(geometry, sess, psi_ket, x_ket, states, ampls)
		'''
		plt.hist(ampls, bins=np.logspace(np.log(ampls.min() + 1e-10) / np.log(10.0), np.log(ampls.max() + 1e-10) / np.log(10.0), num=100), alpha = 0.3, density=True)
		ex_ampls = get_exact_distribution(model)
		plt.hist(ex_ampls, bins=np.logspace(np.log(ampls.min() + 1e-10) / np.log(10.0), np.log(ampls.max() + 1e-10) / np.log(10.0), num=100), alpha = 0.3, weights=ex_ampls, density=True)
		'''
		plt.hist(ampls, bins=np.linspace(ampls.min(), ampls.max(), num=100), alpha = 0.3, density=True)
		ex_ampls = get_exact_distribution(model)
		plt.hist(ex_ampls, bins=np.linspace(ampls.min(), ampls.max(), num=100), alpha = 0.3, weights=ex_ampls, density=True)
		# plt.hist(ampls_check, bins=np.logspace(np.log(ampls.min()) / np.log(10.0), np.log(ampls.max()) / np.log(10.0), num=100), alpha = 0.3)
		# plt.xlim([ampls.min(), ampls.max()])
		# plt.title('accepted = ' + str(accepted))
		# plt.xscale('log')
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
		#grads = get_total_grad_exact(model)
		grads, length = get_total_grad(x_bras, x_kets, H_nms, model)
		# grads = delinearize_gradient(np.real(get_force_exact(model)).astype(np.float32), model)
		# model.variables = [tf.subtract(w_ini, tf.multiply(grad, tf.constant(lr))) for w_ini, grad in zip(model.variables, grads)]
		# print(tf.train.get_or_create_global_step())
		# loss_value = get_current_loss(x_bras, x_kets, H_nms, model)
		# print(get_E_exact(model))
		# print(loss_value, get_E_exact(model))
		# get_force_exact(model)
		# get_check_sample_grad(x_bras, x_kets, H_nms, model)
		loss_value = get_E_exact(model)
		print(loss_value)
		optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())
		logfile.write(str(loss_value) + ' ' + str(best_loss) + ' ' + str(length) + ' ' + str(optimizer._lr.numpy()) + '\n')
		logfile.flush()
		if np.real(loss_value) < best_loss:
			best_loss = np.real(loss_value)
			n_epochs_nonimprove = 0 
		else:
			n_epochs_nonimprove += 1

		#if n_epochs_nonimprove == patience:
		#	lr *= 0.9
		#	n_epochs_nonimprove = 0


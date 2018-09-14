import numpy as np
import tensorflow as tf

def probas(amplitudes):
	return amplitudes[:, 0] ** 2 + amplitudes[:, 1] ** 2

def metropolise_sample_chain(geometry, tf_sess, tf_output, tf_input, 
	                         num_states, len_thermalization, n_parallel_generators = 10):
	states = geometry.get_random_states(n_parallel_generators)

	trajectory_length = len_thermalization + num_states // n_parallel_generators + n_parallel_generators
	trajectory = np.zeros((trajectory_length, states.shape[0], states.shape[1]))
	for n_step in range(trajectory_length):
		trajectory[n_step] = states

		# print(geometry.to_network_format(states).shape)
		amplitudes = tf_sess.run(tf_output, feed_dict = {tf_input : geometry.to_network_format(states)})
		amp_squared = probas(amplitudes)
		# print(amplitudes.shape)
		states_new = geometry.flip_random_spins(states)
		amplitudes_new = tf_sess.run(tf_output, feed_dict = {tf_input : geometry.to_network_format(states_new)})
		amp_squared_new = probas(amplitudes_new)
		accept_probas = np.minimum(np.ones(n_parallel_generators), amp_squared_new / amp_squared)
		accepted = accept_probas > np.random.random(size = n_parallel_generators)
		# print(amp_squared.mean(), accepted.mean())
		if not np.all(~accepted):
			states[accepted, ...] = states_new[accepted, ...]

	states = trajectory[len_thermalization:, ...]
	states = states.reshape((states.shape[0] * states.shape[1], states.shape[2]))
	return states

def sample_nm_pairs(states, geometry, hamiltonian, num_states_rhs):
	x_bras, x_kets, H_nms = [], [], []
	nm_pairs = []  # contains state_n, state_m and matrix element H_{nm}
	for _ in range(num_states_rhs):
		state = states[np.random.randint(low=0, high = states.shape[0])]
		Hstates = hamiltonian(state)
		for Hstate in Hstates:
			x_bras.append(geometry.to_network_format(state)[0, ...])
			x_kets.append(geometry.to_network_format(Hstate[0])[0, ...])
			H_nms.append(Hstate[1])
	return np.array(x_bras), np.array(x_kets), np.array(H_nms)

def sample_n_values(states, geometry, num_states):
	x_bras = []
	for _ in range(num_states):
		x_bras.append(geometry.to_network_format(states[np.random.randint(low=0, high = states.shape[0])]))
	return np.array(x_bras)


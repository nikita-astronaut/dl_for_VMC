import numpy as np
import tensorflow as tf
from tqdm import tqdm

# def probas(amplitudes):
# 	return amplitudes[:, 0] ** 2 + amplitudes[:, 1] ** 2

def metropolise_sample_chain(geometry, tf_sess, tf_output, tf_input, 
	                         num_states, len_thermalization, n_parallel_generators = 100, n_drop = 10):
	states = geometry.get_random_states(n_parallel_generators)
	ampl = np.zeros(states.shape[:-1])
	amplitudes = tf_sess.run(tf_output, feed_dict = {tf_input : geometry.to_network_format(states)})
	amp_squared = np.exp(amplitudes[:, 0]) ** 2

	trajectory = np.zeros((0, states.shape[1]))
	trajectory_ampl = np.zeros(trajectory.shape[:-1])

	step_index = 0
	while trajectory.shape[0] < num_states:
		# print(trajectory.shape[0] * 100.0 / num_states)
		states_new = geometry.flip_random_spins(states)
		amplitudes_new = tf_sess.run(tf_output, feed_dict = {tf_input : geometry.to_network_format(states_new)})
		amp_squared_new = np.exp(amplitudes_new[:, 0]) ** 2
		
		accept_probas = np.minimum(np.ones(n_parallel_generators), amp_squared_new / amp_squared)
		accepted = accept_probas > np.random.random(size = n_parallel_generators)
		
		if not np.all(~accepted):
			states[accepted, ...] = states_new[accepted, ...]
			amplitudes[accepted, ...] = amplitudes_new[accepted, ...]
			amp_squared[accepted, ...] = amp_squared_new[accepted, ...]
		step_index += 1
		if step_index < len_thermalization or step_index % n_drop != 0:
			continue	
		
		trajectory = np.concatenate([trajectory, states], axis = 0)
		trajectory_ampl = np.concatenate([trajectory_ampl, amp_squared], axis = 0)
	return trajectory, trajectory_ampl

def metropolise_sample_chain_threshold(geometry, tf_sess, tf_output, tf_input,
                                       num_states, len_thermalization, n_parallel_generators = 100, n_drop = 10,
									   threshold=1e-8):
    states = geometry.get_random_states(n_parallel_generators)
    ampl = np.zeros(states.shape[:-1])
    amplitudes = tf_sess.run(tf_output, feed_dict = {tf_input : geometry.to_network_format(states)})
    amp_squared = np.exp(amplitudes[:, 0]) ** 2

    trajectory = np.zeros((0, states.shape[1]))
    trajectory_ampl = np.zeros(trajectory.shape[:-1])

    step_index = 0
    while trajectory.shape[0] < num_states:
        # print(trajectory.shape[0] * 100.0 / num_states)
        states_new = geometry.flip_random_spins(states)
        amplitudes_new = tf_sess.run(tf_output, feed_dict = {tf_input : geometry.to_network_format(states_new)})
        amp_squared_new = np.exp(amplitudes_new[:, 0]) ** 2

        accepted = amp_squared_new > threshold
        if not np.all(~accepted):
            states[accepted, ...] = states_new[accepted, ...]
            amplitudes[accepted, ...] = amplitudes_new[accepted, ...]
            amp_squared[accepted, ...] = amp_squared_new[accepted, ...]
        step_index += 1
        if step_index < len_thermalization or step_index % n_drop != 0:
            continue

        trajectory = np.concatenate([trajectory, states], axis = 0)  # we add configuration even if it was rejected
        trajectory_ampl = np.concatenate([trajectory_ampl, amp_squared], axis = 0)  # we add even if it was rejected
    return trajectory, trajectory_ampl

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
	return np.array(x_bras), np.array(x_kets), np.array(H_nms), len(H_nms) * 1.0 / num_states_rhs

def sample_n_values(states, geometry, num_states):
	x_bras = []
	for _ in range(num_states):
		x_bras.append(geometry.to_network_format(states[np.random.randint(low=0, high = states.shape[0])]))
	return np.array(x_bras)


import numpy as np
import tensorflow as tf
from tqdm import tqdm

# def probas(amplitudes):
# 	return amplitudes[:, 0] ** 2 + amplitudes[:, 1] ** 2
'''
def metropolise_sample_chain(geometry, model, num_states, len_thermalization, n_parallel_generators = 100, n_drop = 10):
	states = geometry.get_random_states(n_parallel_generators, sector = 1)
	ampl = np.zeros(states.shape[:-1])
	amplitudes = np.array(model(geometry.to_network_format(states).astype(np.float32)))
	amp_squared = amplitudes[:, 0] ** 2 + amplitudes[:, 1] ** 2
	
	accepts_per_chain = np.zeros((0, n_parallel_generators))
	ampls_per_chain = np.zeros((0, n_parallel_generators))

	trajectory = np.zeros((0, states.shape[0], states.shape[1]))
	trajectory_ampl = np.zeros((0, states.shape[0]))

	step_index = 0
	pbar = tqdm(total = num_states)
	while trajectory.shape[0] * trajectory.shape[1] < num_states:
		# print(trajectory.shape[0] * 100.0 / num_states)
		states_new = geometry.flip_random_spins(states)
		amplitudes_new = np.array(model(geometry.to_network_format(states_new).astype(np.float32)))
		amp_squared_new = amplitudes_new[:, 0] ** 2 + amplitudes_new[:, 1] ** 2
		
		accept_probas = np.minimum(np.ones(n_parallel_generators), amp_squared_new / amp_squared)
		accepted = accept_probas > np.random.random(size = n_parallel_generators)
		if not np.all(~accepted):
			states[accepted, ...] = states_new[accepted, ...]
			amplitudes_new[accepted, ...] = amplitudes[accepted, ...]
			amp_squared[accepted, ...] = amp_squared_new[accepted, ...]
		step_index += 1
		if step_index < len_thermalization or step_index % n_drop != 0:
			continue	
		
		trajectory = np.concatenate([trajectory, states[np.newaxis, ...]], axis = 0)
		trajectory_ampl = np.concatenate([trajectory_ampl, amp_squared[np.newaxis, ...]], axis = 0)
		accepts_per_chain = np.concatenate([accepts_per_chain, accepted[np.newaxis, ...]], axis = 0)
		ampls_per_chain = np.concatenate([ampls_per_chain, amp_squared[np.newaxis, ...]], axis = 0)
		pbar.update(trajectory.shape[1] * trajectory.shape[0])
	pbar.close()

	good_trajectories = np.mean(accepts_per_chain, axis = 0) > -0.10
	trajectory = trajectory[:, good_trajectories, :]
	trajectory_ampl = trajectory_ampl[:, good_trajectories]
	return trajectory.reshape((-1, states.shape[1])), trajectory_ampl.reshape((-1)), np.mean(accepts_per_chain, axis = 0)[good_trajectories], np.mean(ampls_per_chain, axis = 0)[good_trajectories]
'''

def metropolise_sample_chain(geometry, model, num_states, len_thermalization, n_drop = 10):
	state = geometry.get_random_states(1, sector = 1)[0]
	amplitude = model(geometry.to_network_format(state[np.newaxis, ...]).astype(np.float32))[0].numpy()
	amp_squared = np.exp(amplitude[0]) ** 2

	trajectory = np.zeros((0, state.shape[0]))
	trajectory_ampl = np.zeros((0))
	accepts = []
	step_index = 0
	pbar = tqdm(total = num_states + len_thermalization)
	while trajectory.shape[0] < num_states:
		state_new = geometry.flip_random_spins(state[np.newaxis, ...])[0]
		amplitude_new = model(geometry.to_network_format(state_new[np.newaxis, ...]).astype(np.float32))[0].numpy()
		amp_squared_new = np.exp(amplitude_new[0]) ** 2

		accept_probas = np.min([1.0, amp_squared_new / amp_squared])
		accepted = accept_probas > np.random.random()
		if amp_squared_new / amp_squared > np.exp(5.5) ** 2:
			pbar.n = 0
			pbar.refresh()
			trajectory = np.zeros((0, state.shape[0]))
			print('high rise happened!')
			continue
		if accepted:
			state = state_new
			amp_squared = amp_squared_new

		if step_index >= len_thermalization:
			accepts.append(accepted)
			trajectory = np.concatenate([trajectory, state[np.newaxis, ...]], axis = 0)
			trajectory_ampl = np.concatenate([trajectory_ampl, np.array([amp_squared])], axis = 0) 
		step_index += 1
		pbar.update(1)
	pbar.close()
	print('acceptance rate =', np.mean(accepts))
	return trajectory, trajectory_ampl


def metropolise_check(geometry, tf_sess, tf_output, tf_input,
                      states, amp_squared):

    states_new = geometry.flip_random_spins(states)
    amplitudes_new = tf_sess.run(tf_output, feed_dict = {tf_input : geometry.to_network_format(states_new)})
    amp_squared_new = np.exp(amplitudes_new[:, 0]) ** 2

    accept_probas = np.minimum(np.ones(amp_squared.shape[0]), amp_squared_new / amp_squared)
    accepted = accept_probas > np.random.random(size = amp_squared.shape[0])

    if not np.all(accepted):
        states_new[~accepted, ...] = states[~accepted, ...]
        amp_squared_new[~accepted, ...] = amp_squared[~accepted, ...]
    
    return states_new, amp_squared_new, np.mean(accepted)

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
	
	for state in states:
		Hstates = hamiltonian(state)
		x_kets_this = []
		H_nms_this = []
		x_bras.append(geometry.to_network_format(state))#[0, ...])

		for Hstate in Hstates:
			x_kets_this.append(geometry.to_network_format(Hstate[0]))#[0, ...])
			H_nms_this.append(Hstate[1])

		x_kets.append(np.array(x_kets_this))
		H_nms.append(np.array(H_nms_this))
	return x_bras, x_kets, H_nms

def sample_n_values(states, geometry, num_states):
	x_bras = []
	for _ in range(num_states):
		x_bras.append(geometry.to_network_format(states[np.random.randint(low=0, high = states.shape[0])]))
	return np.array(x_bras)


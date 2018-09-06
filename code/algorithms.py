import numpy as np
import tensorflow as tf

def probas(amplitudes):
	return amplitudes[:, 0] ** 2 + amplitudes[:, 1] ** 2

def metropolise_sample_chain(geometry, tf_sess, tf_output, tf_input, 
	                         n_parallel_generators = 100, trajectory_length = 100):
	states = geometry.get_random_states(n_parallel_generators)

	trajectory = np.zeros((trajectory_length, states.shape[0], states.shape[1]))
	for n_step in range(trajectory_length):
		trajectory[n_step] = states

		amplitudes = tf_sess.run(inference, feed_dict = {tf_input : states})
		amp_squared = probas(amplitudes)

		states_new = geometry.flip_random_spins(states)
		amplitudes_new = tf_sess.run(inference, feed_dict = {tf_input : states_new})
		amp_squared_new = probas(amplitudes_new)
		accept_probas = np.minimum(np.ones(n_parallel_generators), amp_squared_new / amp_squared_old)
		accepted = accept_probas > np.random.random(size = n_parallel_generators)

		if not np.all(~accepted):
			states[:, accepted] = states_new[:, accepted]

	return trajectory
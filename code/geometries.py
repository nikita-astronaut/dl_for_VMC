from copy import deepcopy
import numpy as np

class square_geometry:
	def __init__(self, shape):
		self.Lx, self.Ly = shape

	def _to_xy(self, global_index):
		return global_index % self.Lx, global_index // self.Lx

	def _to_global(self, x, y):
		return y * self.Lx + x

	def to_network_format(self, wave_functions):
		return wave_functions.reshape((-1, self.Lx, self.Ly, 1))
		'''
		wave_function_2D = np.zeros((Lx, Ly))
		for global_index, amplitude in enumerate(wave_function):
			x, y = _to_global(x, y)
			wave_function_2D[x, y] = amplitude
		return wave_function_2D[..., np.newaxis]
		'''
	def get_random_states(self, n_states):  # returns multiple states at once
		return np.random.choice(np.array([-1.0, 1.0]), size = (n_states, self.Lx * self.Ly * 1))

	def flip_random_spins(self, states):  # flips random spins in all states at once
		new_state = deepcopy(states)
		new_state[np.arange(states.shape[0]), np.random.randint(low=0, high = states.shape[1], size = states.shape[0])] *= -1.0
		return new_state

	def __call__(self, global_index):
		'''
			by global_index, computes global_indexes of neighboring sites
        '''
		x, y = self._to_xy(global_index)
		# return [_to_global(x + 1, y), _to_global(x - 1, y), _to_global(x, y + 1), _to_global(x, y - 1)]
		return [self._to_global((x + 1) % self.Lx, y), self._to_global(x, (y + 1) % self.Ly)]  # to avoid double-counting, we only cound neighbors in forward direction
                                                             # this is valid ONLY if lattice is periodic, so all pairs will effectively appear

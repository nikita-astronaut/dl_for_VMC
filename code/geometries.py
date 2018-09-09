from copy import deepcopy

class square_geometry:
	def __init__(self, shape):
		self.Lx, self.Ly = shape

	def __call__(self, global_index):
		'''
			by global_index, computes global_indexes of neighboring sites
		'''
		x, y = _to_xy(global_index)
		# return [_to_global(x + 1, y), _to_global(x - 1, y), _to_global(x, y + 1), _to_global(x, y - 1)]
		return [_to_global(x + 1, y), _to_global(x, y + 1)]  # to avoid double-counting, we only cound neighbors in forward direction
															 # this is valid ONLY if lattice is periodic, so all pairs will effectively appear
	def _to_xy(self, global_index):
		return global_index % self.Lx, global_index // self.Lx

	def _to_global(self, x, y):
		return y * Lx + x

	def to_network_format(self, wave_function):
		wave_function_2D = np.zeros((Lx, Ly))
		for global_index, amplitude in enumerate(wave_function):
			x, y = _to_global(x, y)
			wave_function_2D[x, y] = amplitude
		return wave_function_2D

	def get_random_state(self, n_states):  # returns multiple states at once
		return np.random.choice(np.array([-1.0, 1.0]), size = (self.Lx * self.Ly, n_states))

	def flip_random_spins(self, states):  # flips random spins in all states at once
		new_state = deepcopy(states)
		new_state[np.random.randint(low=0, high = states.shape[0], size = states.shape[1]), np.arange(states.shape[1])] *= -1.0
		return new_state
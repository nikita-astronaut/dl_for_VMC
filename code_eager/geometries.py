from copy import deepcopy
import numpy as np

class square_geometry:
	def __init__(self, shape):
		self.Lx, self.Ly = shape

	def _to_xy(self, global_index):
		return global_index % self.Lx, global_index // self.Lx

	def _to_global(self, x, y):
		return y * self.Lx + x
	
	#def to_network_format(self, wave_functions):
	#	return wave_functions.reshape((self.Lx, self.Ly, 1))
	def to_network_format(self, wave_function):
		return wave_function
	
	def get_random_states(self, n_states, sector=None):  # returns multiple states at once
		if sector == None:
			return np.random.choice(np.array([-1.0, 1.0]), size = (n_states, self.Lx * self.Ly * 1))
		states = np.zeros((0, self.Lx * self.Ly))
		for _ in range(n_states):
			if sector > 0:
				state = np.ones(self.Lx * self.Ly) * (-1.0)
				state[np.random.choice(self.Lx * self.Ly, self.Lx * self.Ly // 2 + sector, replace=False)] *= -1.0
			else:
				state = np.ones(self.Lx * self.Ly) * (1.0)
				state[np.random.choice(self.Lx * self.Ly, self.Lx * self.Ly // 2 - sector, replace=False)] *= -1.0
			states = np.concatenate([states, state[np.newaxis, ...]], axis = 0)
		return states

	def flip_random_spins(self, states):  # flips random spins in all states at once
		new_states = deepcopy(states)

		for i in range(new_states.shape[0]):
			flipped = False
			while not flipped:
				spins = np.random.choice(self.Lx * self.Ly, 2, replace=False)
				if np.prod(new_states[i, spins]) < 0:
					new_states[i, spins] *= -1.0
					flipped = True

		return new_states

	def get_all_states(self, sector=1):
		states = []
		for n in range(2 ** (self.Lx * self.Ly)):
			binary = np.binary_repr(n, width = self.Lx * self.Ly)
			binary = np.array([int(x) for x in binary]) * 2.0 - 1.0
			if np.sum(binary) == sector:
				states.append(binary)
		return np.array(states)#self.to_network_format(np.array(states))

	def __call__(self, global_index):
		'''
			by global_index, computes global_indexes of neighboring sites
        '''
		x, y = self._to_xy(global_index)
		# return [_to_global(x + 1, y), _to_global(x - 1, y), _to_global(x, y + 1), _to_global(x, y - 1)]
		return [self._to_global((x + 1) % self.Lx, y), self._to_global(x, (y + 1) % self.Ly)]  # to avoid double-counting, we only cound neighbors in forward direction
                                                             # this is valid ONLY if lattice is periodic, so all pairs will effectively appear

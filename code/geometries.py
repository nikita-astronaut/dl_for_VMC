class square_geometry:
	def __init__(self, Lx, Ly):
		self.Lx = Lx
		self.Ly = Ly

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

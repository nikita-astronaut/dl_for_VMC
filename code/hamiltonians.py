from copy import deepcopy


class AFH:
	def __init__(self, geometry):
		self.geometry = geometry

	def __call__(self, state):
		action_result = []  # stores [(state_0, amplitude_0), (state_1, amplitude_1), ...]
		initial_state_amplitude = 0.0

		for global_index in range(state.shape[0]):
			neighbors = geometry(global_index)
			
			for neighbor in neighbors:
				s_i = state[global_index]
				s_j = state[neighbor]

				if s_i == s_j:
					initial_state_amplitude += 1.0
				else:
					initial_state_amplitude -= 1.0
					new_state = deepcopy(state)
					new_state[global_index] *= -1.0
					new_state[neighbor] *= -1.0
					action_result.append((new_state, 2.0))
		action_result.append((state, initial_state_amplitude))

		return action_result

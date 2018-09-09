from models import conv2d_model
import numpy as np

model = conv2d_model
epochs = 1000
num_nm_rhs = 200  # required for <H> estimation
num_n_samples = 1000  # required for -\log <psi>^2 estimation
random_seed = 42
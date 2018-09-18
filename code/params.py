from models import conv2d_model
import numpy as np
from geometries import square_geometry
from hamiltonians import AFH

model = conv2d_model
epochs = 1000
epoch_size = 5000
num_nm_rhs = 1000  # required for <H> estimation
num_n_samples = 100000  # required for -\log <psi>^2 estimation
random_seed = 42
input_shape = (5, 5)
hamiltonian = AFH
geometry = square_geometry
len_thermalization = 100
lr = 1e-3
n_parallel = 100
n_drop = 10
n_epoch_passes = 3

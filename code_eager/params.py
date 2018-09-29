from models import conv2d_model
import numpy as np
from geometries import square_geometry
from hamiltonians import AFH

model = conv2d_model
epochs = 1000
epoch_size = 500
n_minibatches = 5
num_nm_rhs = 500  # required for <H> estimation
num_n_samples = 100000  # required for -\log <psi>^2 estimation
random_seed = 42
input_shape = (3, 3)
hamiltonian = AFH
geometry = square_geometry
len_thermalization = 500
lr = 3e-2
n_parallel = 1
n_drop = 5
n_epoch_passes = 1
patience = 30

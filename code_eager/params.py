from models import conv2d_model, dense_model
import numpy as np
from geometries import square_geometry
from hamiltonians import AFH

model = dense_model
epochs = 100000
epoch_size = 1000
n_minibatches = 5
num_nm_rhs = 1000  # required for <H> estimation
num_n_samples = 100000  # required for -\log <psi>^2 estimation
random_seed = 42
input_shape = (3, 3)
hamiltonian = AFH
geometry = square_geometry
len_thermalization = 1000
lr = 3e-2
n_parallel = 1
n_drop = 5
n_epoch_passes = 1
patience = 30

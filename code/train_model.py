import params
import numpy as np
import tensorflow as tf

model = params.model
epochs = params.epochs
num_nm_rhs = params.num_nm_rhs  # required for <H> estimation
num_n_samples = params.num_n_samples  # required for -\log <psi>^2 estimation


Hnm = tf.placeholder("float", [None])  # placeholder for H matrix elements
psi_bra = tf.placeholder("float", [None, 2])  # placeholder for <psi| values (real + im)
psi_ket = tf.placeholder("float", [None, 2])  # placeholder for |psi> values (real + im)
loss_H = tf.reduce_mean(Hnm * psi_bra[:, 0] * psi_ket[:, 0]) + tf.reduce_mean(Hnm * psi_bra[:, 1] * psi_ket[:, 1])
loss_psi = -tf.log(0.5 * tf.reduce_mean(psi_bra[:, 0] * psi_bra[:, 0] + psi_bra[:, 1] * psi_bra[:, 0]) + 
	               0.5 * tf.reduce_mean(psi_ket[:, 0] * psi_ket[:, 0] + psi_ket[:, 1] * psi_ket[:, 0]))
loss = loss_H + loss_psi

# %%
import tensorflow as tf
from tensorflow import math as tfm


class IsingParams:
    def __init__(self, L=32, pbc=False, lattice="square", J=1.,
                 next_nearest=False):
        self.L = L
        self.pbc = pbc
        self.lattice = lattice
        self.J = J
        self.next_nearest = next_nearest


def energy(sample, params):
    """Calculates energy assuming open boundary conditions

    Args:
        sample (tf.Tensor): A batch of Ising lattices sampled from a VAN network
        params (IsingParams): An object defining various parameters of the Ising model
    """
    if params.pbc:
        # Adding nearest neighbours along y
        term = tf.roll(sample, 1, 1) * sample
        E = tfm.reduce_sum(term, axis=[1, 2, 3])
        # Adding nearest neighbours along x
        term = tf.roll(sample, 1, 2) * sample
        E += tfm.reduce_sum(term, axis=[1, 2, 3])
        if params.lattice == 'tri':
            term = tf.roll(sample, [1, 1], [1, 2]) * sample
            E += tfm.reduce_sum(term, axis=[1, 2, 3])
    else:
        # Adding nearest neighbours along y
        term = sample[:, :-1, :, :] * sample[:, 1:, :, :]
        E = tfm.reduce_sum(term, axis=[1, 2, 3])
        # Adding nearest neighbours along x
        term = sample[:, :, :-1, :] * sample[:, :, 1:, :]
        E += tfm.reduce_sum(term, axis=[1, 2, 3])
        if params.lattice == 'tri':
            term = sample[:, :-1, :-1, :] * sample[:, 1:, 1:, :]
            E += tfm.reduce_sum(term, axis=[1, 2, 3])
    return tf.cast(params.J * E, tf.float32)

# %%

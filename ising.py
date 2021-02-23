# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow import math as tfm

# %%
J = -1.#This is the value of Ising coupling constant

# %%
def energy(sample):
    """Calculates energy assuming open boundary conditions

    Args:
        sample (tf.Tensor): A batch of Ising lattices sampled from a VAN network
    """
    #Adding nearest neighbours along x
    term = sample[:,:-1,:,:]*sample[:,1:,:,:]
    energy = tfm.reduce_sum(term, axis=[1,2,3])
    #Adding nearest neighbours along y
    term = sample[:,:,:-1,:]*sample[:,:,1:,:]
    energy += tfm.reduce_sum(term, axis=[1,2,3])
    return tf.cast(J*energy, tf.float32)


# %%

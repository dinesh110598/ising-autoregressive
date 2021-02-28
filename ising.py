# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow import math as tfm

# %%
class AutoregressiveModel(tfk.Model):
    def __init__(self, L, epsilon):
        super(AutoregressiveModel, self).__init__()
        self.L = L
        self.epsilon = tf.cast(epsilon, tf.float32)

        self.x_hat_mask = np.ones([1, self.L, self.L, 1], np.float32)
        self.x_hat_mask[:,0,0,:] = 0
        self.x_hat_bias = np.zeros([1, self.L, self.L, 1], np.float32)
        self.x_hat_bias[:,0,0,:] = 0.5
        
    def sample(self, batch_size):
        sample = np.zeros([batch_size,self.L,self.L,1], np.float32)
        for i in range(self.L):
            for j in range(self.L):
                x_hat = self.call(sample)
                sample[:,i,j,:] = np.random.binomial(1, 
                                x_hat[:,i,j,:], [batch_size,1])*2 - 1 #type: ignore
        #x_hat = self.call(sample)
        flip = np.random.binomial(1, 0.5, [batch_size,1,1,1])*2 - 1
        sample = sample*flip#Done to ensure Z2 symmetry 
        return sample
    
    def graph_sampler(self, sample, seed):
        #Same as sample method above but specialised for graph compilation
        batch_size = sample.shape[0]
        counts = tf.ones([batch_size, 1])
        sample.assign(tf.zeros(sample.shape))
        tf_binomial = tf.random.stateless_binomial
        for i in range(self.L):
            for j in range(self.L):
                seed.assign((seed*1664525 + 1013904223) % 2**31)
                x_hat = self.call(sample)
                sample = sample[:,i,j,:].assign(tf_binomial([batch_size,1], seed, 
                                counts, x_hat[:,i,j,:], tf.float32)*2 - 1) #type: ignore
        #x_hat = self.call(sample)
        seed.assign((seed*1664525 + 1013904223) % 2**31)
        counts = tf.expand_dims(counts, -1)
        counts = tf.expand_dims(counts, -1)
        flip = tf_binomial([batch_size,1,1,1], seed, counts, 0.5*counts, 
            tf.float32)*2 - 1
        sample.assign(sample*flip)#Done to ensure Z2 symmetry 
        return sample
    
    def _log_prob(self, sample, x_hat):
        mask = (sample + 1)/2#Remember that x_hat gives prob of all 1's not given sample's
        log_prob = (
            tfm.log(x_hat + self.epsilon)*tf.cast(mask, tf.float32) + #type: ignore
            tfm.log(1 - x_hat + self.epsilon)*tf.cast(1 - mask, tf.float32))#type: ignore
        log_prob = tfm.reduce_sum(log_prob, [1,2,3])
        return log_prob
    
    def log_prob(self, sample):
        x_hat = self.call(sample)
        log_prob = self._log_prob(sample, x_hat)
        #We'll average prob of original sample to that of inverses to enforce Z2
        sample_inv = -sample
        x_hat_inv = self.call(sample_inv)
        log_prob_inv = self._log_prob(sample_inv, x_hat_inv)
        log_prob = tfm.reduce_logsumexp(
            tf.stack([log_prob, log_prob_inv]),
            axis=0)
        log_prob -= tfm.log(2.)
        return tf.cast(log_prob, tf.float32)

J = -1.#This is the value of Ising coupling constant

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

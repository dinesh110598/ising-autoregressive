# %%
import ising
import numpy as np
from tensorflow import keras as tfk
from tensorflow import math as tfm
import tensorflow as tf
from tqdm import tqdm
from time import time
# %%
class Trainer:
    def __init__(self, model, batch_size=50, learning_rate=0.01):
        self.lr_schedule = tfk.optimizers.schedules.ExponentialDecay(0.01, 500, 0.4, True)
        self.optimizer = tfk.optimizers.Adam(self.lr_schedule, 0.5, 0.999)
        self.beta_anneal = 0.99
        self.model = model
        self.batch_size= 50
        self.sample_graph = tf.Variable(tf.zeros([batch_size, self.model.L, self.model.L, 1]), 
                                        trainable=False)

    @tf.function
    def backprop(self, beta):
        """Performs backpropagation on the calculated loss function

        Args:
            beta (float): Inverse temperature

        Returns:
            loss (float): The current loss function for the sampled batch
        """
        self.sample_graph = self.model.graph_sampler(self.sample_graph, self.seed)
        energy = ising.energy(self.sample_graph)
        beta = tf.cast(beta, tf.float32)
        with tf.GradientTape(True, False) as tape:
            tape.watch(self.model.trainable_weights)
            log_prob = self.model.log_prob(self.sample_graph)
            with tape.stop_recording():
                loss = (log_prob + beta*energy) / (self.model.L**2)#type: ignore
            loss_reinforce = tfm.reduce_mean((loss - tfm.reduce_mean(loss))*log_prob)
        grads = tape.gradient(loss_reinforce, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        print("Tracing")
        return loss/beta, energy

    def train_loop(self, iter, beta, anneal=True):
        """Runs the unsupervised training loop for VAN training.

        Args:
            iter (int): No of batches to use for training
            batch_size (int): No of lattices to sample for single training step
            beta (float): Inverse temperature to use
        Options:
            If model is not None, **kwargs maybe supplied to initialize it.
            See docstring for ising.PixelCNN() for details.
        """
        
        beta = tf.convert_to_tensor(beta)
        beta_conv = tf.cast(beta, tf.float32)
        history = {'step':[],'Free energy mean':[], 'Free energy std':[], 'Energy mean':[], 'Energy std':[],
        'Train time':[]}
        interval = 20

        #Routines required for graph compilation
        self.seed = tf.Variable(np.random.randint(-2**30, 2**30, size=2, dtype=np.int32),
            dtype=tf.int32, trainable=False)
        t1 = time()
        
        for step in tqdm(range(iter)):
            if anneal==True:
                beta = beta_conv*(1 - self.beta_anneal**step)
            loss, energy = self.backprop(beta, self.seed) #type: ignore

            if (step%interval) == interval-1:
                t2 = time()
                history['step'].append(step+1)
                history['Free energy mean'].append( tfm.reduce_mean(loss))
                history['Free energy std'].append( tfm.reduce_std(loss))
                history['Energy mean'].append( tfm.reduce_mean(energy))
                history['Energy std'].append( tfm.reduce_std(energy))
                history['Train time'].append( (t2-t1)/interval)
                t1 = time()
        
        return history
# %%
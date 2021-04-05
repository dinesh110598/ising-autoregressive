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
    def __init__(self, model, batch_size=50, learning_rate=0.001):
        self.lr_schedule = tfk.optimizers.schedules.ExponentialDecay(learning_rate, 200, 0.9, True)
        self.optimizer = tfk.optimizers.Adam(self.lr_schedule, 0.5, 0.999)
        self.beta_anneal = 0.99
        self.model = model
        self.batch_size= 50
        self.sample_graph = tf.Variable(tf.zeros([batch_size, self.model.L, self.model.L, 1]), 
                                        trainable=False)
        self.seed = tf.Variable(np.random.randint(-2**30, 2**30, size=2, dtype=np.int32),
                                dtype=tf.int32, trainable=False)

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
            #regularizer = tfm.reduce_euclidean_norm(self.model(self.sample_graph) +
                                                #self.model(-self.sample_graph)-1)
            #regularizer = tfm.divide(regularizer, self.model.L**2)
            loss_reinforce = tfm.reduce_mean((loss - tfm.reduce_mean(loss))*log_prob)
            #loss_reinforce = tfm.add(loss_reinforce, regularizer)
        grads = tape.gradient(loss_reinforce, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss/beta, energy

    def train_loop(self, iter, beta, anneal=True):
        
        beta = tf.convert_to_tensor(beta, tf.float32)
        beta_conv = tf.cast(beta, tf.float32)
        history = {'step':[],'Free energy mean':[], 'Free energy std':[], 'Energy mean':[], 'Energy std':[],
        'Train time':[]}
        interval = 20

        t1 = time()
        
        for step in tqdm(range(iter)):
            if anneal==True:
                beta = beta_conv*(1 - self.beta_anneal**step)
            loss, energy = self.backprop(beta) #type: ignore

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

    @tf.function
    def var_backprop(self, beta):
        sample = self.model.graph_sampler(self.batch_size, self.seed, beta)
        energy = ising.energy(sample)
        beta = tf.cast(beta, tf.float32)
        with tf.GradientTape(True, False) as tape:
            tape.watch(self.model.trainable_weights)
            log_prob = self.model.log_prob(sample, beta)
            with tape.stop_recording():
                loss = (log_prob + beta*energy) / (self.model.L**2)#type: ignore
            #regularizer = tfm.reduce_euclidean_norm(self.model(self.sample_graph, beta) +
                                                #self.model(-self.sample_graph, beta) - 1)
            #regularizer = tfm.divide(regularizer, self.model.L**2)
            loss_reinforce = tfm.reduce_mean((loss - tfm.reduce_mean(loss))*log_prob)
            #loss_reinforce = tfm.add(loss_reinforce, regularizer)
        grads = tape.gradient(loss_reinforce, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss/beta, energy

    def var_train_loop(self, iter, anneal=True, mean=0.5, delta=0.1):
        history = {'step': [], 'Free energy mean': [], 'Free energy std': [], 'Energy mean': [], 'Energy std': [],
                   'Train time': []}
        interval = 20
        t1 = time()

        for step in tqdm(range(iter)):
            if anneal==True:
                mean_beta = mean*(1 - self.beta_anneal**step)
            else:
                mean_beta = mean
            beta = tf.random.normal([], mean_beta, delta)
            loss, energy = self.var_backprop(beta)  # type: ignore

            if (step % interval) == interval-1:
                t2 = time()
                history['step'].append(step+1)
                history['Free energy mean'].append(tfm.reduce_mean(loss))
                history['Free energy std'].append(tfm.reduce_std(loss))
                history['Energy mean'].append(tfm.reduce_mean(energy))
                history['Energy std'].append(tfm.reduce_std(energy))
                history['Train time'].append((t2-t1)/interval)
                t1 = time()

        return history
# %%

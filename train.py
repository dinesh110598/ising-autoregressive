# %%
from tensorflow.keras import layers
import ising
import layers
import numpy as np
from tensorflow import keras as tfk
from tensorflow import math as tfm
import tensorflow as tf
from tqdm import tqdm
from time import time
# %%
lr_schedule = tfk.optimizers.schedules.ExponentialDecay(0.01, 500, 0.8, True)
optimizer = tfk.optimizers.Adam(lr_schedule, 0.5, 0.999)
beta_anneal = 0.99
net = layers.PixelCNN()

def backprop(beta, sample, seed):
    """Performs backpropagation on the calculated loss function

    Args:
        beta (float): Inverse temperature

    Returns:
        loss (float): The current loss function for the sampled batch
    """
    sample = net.graph_sampler(sample, seed)
    energy = ising.energy(sample)
    beta = tf.cast(beta, tf.float32)
    with tf.GradientTape(True, False) as tape:
        tape.watch(net.trainable_weights)
        log_prob = net.log_prob(sample)
        with tape.stop_recording():
            loss = (log_prob + beta*energy) / (net.L**2)#type: ignore
        loss_reinforce = tfm.reduce_mean((loss - tfm.reduce_mean(loss))*log_prob)
    grads = tape.gradient(loss_reinforce, net.trainable_weights)
    optimizer.apply_gradients(zip(grads, net.trainable_weights))
    return loss, energy

backprop_graph = tf.function(backprop)#Constructs a graph for faster gradient calculations

def train_loop(iter, batch_size, beta, anneal=True, **kwargs):
    """Runs the unsupervised training loop for VAN training.

    Args:
        iter (int): No of batches to use for training
        batch_size (int): No of lattices to sample for single training step
        beta (float): Inverse temperature to use
    Options:
        If net is not None, **kwargs maybe supplied to initialize it.
        See docstring for ising.PixelCNN() for details.
    """
    
    beta_conv = tf.cast(beta, tf.float32)
    history = {'step':[],'Free energy mean':[], 'Free energy std':[], 'Energy mean':[], 'Energy std':[],
    'Train time':[]}
    interval = 20

    #Routines required for graph compilation
    sample_graph = tf.Variable(tf.zeros([batch_size,net.L,net.L,1]), trainable=False)
    seed_graph = tf.Variable(np.random.randint(-2**30, 2**30, size=2, dtype=np.int32),
        dtype=tf.int32, trainable=False)
    t1 = time()
    
    for step in tqdm(range(iter)):
        if anneal==True:
            beta = beta_conv*(1 - beta_anneal**step)
        loss, energy = backprop_graph(beta, sample_graph, seed_graph) #type: ignore

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
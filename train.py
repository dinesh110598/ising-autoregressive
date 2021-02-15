import ising
from tensorflow import keras as tfk
from tensorflow import math as tfm
import tensorflow as tf
from tqdm import tqdm
from time import time
# %%
learning_rate = tf.Variable(0.001, trainable=False, dtype=tf.float32)
optimizer = tfk.optimizers.Adam(learning_rate, 0.5, 0.999)
beta_anneal = 0.99

def train_loop(iter, batch_size, beta, net=None, anneal=True, **kwargs):
    """Runs the unsupervised training loop for VAN training.

    Args:
        iter (int): No of batches to use for training
        batch_size (int): No of lattices to sample for single training step
        beta (float): Inverse temperature to use
        net (tf.keras.Model): Pre-initialized model to train. If None, 
            model will be initialized in this function
    Options:
        If net is not None, **kwargs maybe supplied to initialize it.
        See docstring for ising.PixelCNN() for details.
    """
    if net==None:
        net = ising.PixelCNN(**kwargs)
    #step_schedule = 0
    #loss_schedule = tf.TensorArray(tf.float32, 0, True, False)
    #We'll reduce the learning rate by a factor, if loss doesn't reduce significantly
    #after many batches of training
    beta_conv = tf.cast(beta, tf.float32)
    history = {'step':[],'Free energy mean':[], 'Free energy std':[], 'Energy mean':[], 'Energy std':[],
    'Train time':[]}
    interval = 20

    def backprop(beta, sample):
        """Performs backpropagation on the calculated loss function

        Args:
            beta (float): Inverse temperature

        Returns:
            loss (float): The current loss function for the sampled batch
        """
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
    t1 = time()
    
    for step in tqdm(range(iter)):
        if anneal==True:
            beta = beta_conv*(1 - beta_anneal**step)
        sample = net.sample(batch_size)
        loss, energy = backprop_graph(beta, sample) #type: ignore
        #We schedule to decrease the learning rate when loss doesn't reduce signifcantly
        #loss_schedule = loss_schedule.write(step_schedule, tfm.reduce_mean(loss))
        #diff = 0.01
        #if step_schedule > 500:
            #diff = (
                #tfm.reduce_mean(loss_schedule.gather(tf.range(50))) - 
                #tfm.reduce_mean(loss_schedule.gather(tf.range(step_schedule-50, step_schedule)))
                #)
        #step_schedule += 1
        #if diff > 0 and diff < 1e-4:
            #learning_rate.assign(learning_rate*0.92)
            #step_schedule = 0

        if (step%interval) == interval-1:
            t2 = time()
            history['step'].append(step+1)
            history['Free energy mean'].append( tfm.reduce_mean(loss))
            history['Free energy std'].append( tfm.reduce_std(loss))
            history['Energy mean'].append( tfm.reduce_mean(energy))
            history['Energy std'].append( tfm.reduce_std(energy))
            history['Train time'].append( (t2-t1)/interval)
            t1 = time()
    #loss_schedule = loss_schedule.mark_used()
    
    return history

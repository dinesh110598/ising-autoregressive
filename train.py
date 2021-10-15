# %%
import ising
from tensorflow import keras as tfk
from tensorflow import math as tfm
import tensorflow as tf
from BPnet import BPnet
from tqdm import tqdm


@tf.function
def backpropagate(x, beta, model, pars, opt):
    """Performs backpropagation on the calculated loss function

    Args:
        x (tf.Tensor): Sampled Ising spin configuration
        beta (float): Inverse temperature
        model (BPnet): Autoregressive neural network that samples and calculates the logits of Ising Model
        pars (ising.IsingParams): Object defining various parameters of the Ising model
        opt (tfk.optimizers.Optimizer): Neural network optimizer object

    Returns:
        loss (float): The current loss function for the sampled batch
    """
    E = ising.energy(x, pars)
    with tf.GradientTape(False, False) as tape:
        tape.watch(model.trainable_weights)
        log_prob = model.log_prob(x)
        with tape.stop_recording():
            loss = (log_prob + beta * E) / (pars.L ** 2)
        loss_reinforce = tfm.reduce_mean((loss - tfm.reduce_mean(loss)) * log_prob)
    grads = tape.gradient(loss_reinforce, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))
    return loss / beta, E


def train_loop(model, tot_steps, beta, pars, batch_size=100, anneal=False, init_eta=0.005,
               beta_anneal=0.95):
    lr_schedule = tfk.optimizers.schedules.ExponentialDecay(init_eta, 250, 0.9, staircase=True)
    opt = tfk.optimizers.Adam(lr_schedule)
    beta_conv = beta

    outer = tqdm(total=tot_steps, desc='Training steps', position=0)
    F_log = tqdm(total=0, position=1, bar_format='{desc}')
    for step in range(tot_steps):
        if anneal:
            beta = beta_conv * (1 + beta_anneal**(step+1))
        x = model.sample(pars.L, batch_size)
        beta = tf.constant(beta, tf.float32)
        F, E = backpropagate(x, beta, model, pars, opt)
        mean_F = tfm.reduce_mean(F)
        std_F = tfm.reduce_std(F)
        outer.update(1)
        F_log.set_description_str(f'Average F: {mean_F} \t Std F: {std_F}')
        # Saving weights once in a while
        if (step+1) % 500 == 0:
            model.save_weights(f'Saves/Chkpts/b{beta_conv}_s{step+1}')


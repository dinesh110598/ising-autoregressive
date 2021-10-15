import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow import math as tfm
import tensorflow_addons as tfa


class ConvBlock(tfk.layers.Layer):
    def __init__(self, p, n, mask='B', last_layer=False, **kwargs):
        super().__init__(**kwargs)
        if mask == 'A':
            self.res = 0
        else:
            self.res = 1
        r = self.res
        self.n = n
        self.last_layer = last_layer

        self.l_conv = [tfk.layers.Conv2D(p, (1, n - 1 + r), activation=tfa.activations.mish),
                       tfk.layers.Conv2D(p, 1, activation=tfa.activations.mish)]
        self.r_conv = [tfk.layers.Conv2D(p, (1, n), activation=tfa.activations.mish),
                       tfk.layers.Conv2D(p, 1, activation=tfa.activations.mish)]
        self.m_conv = [tfk.layers.Conv2D(p, 1, activation=tfa.activations.mish, use_bias=False),
                       tfk.layers.Conv2D(p, 1, activation=tfa.activations.mish)]
        if r:
            self.m_conv.append(tfk.layers.Conv2D(p, 1, activation=tfa.activations.mish, use_bias=False))
        else:
            self.l_conv[0].build([None, None, 1])
            self.r_conv[0].build([None, None, 1])
            self.m_conv[0].build([None, None, 1])

    def call(self, x_l, x_m, x_r):  # Overriding parent method is what we seek
        if self.res == 1:
            x_m2 = x_m
        else:
            x_l = tfk.layers.Cropping2D(((0, 0), (0, 1)))(x_l)
            x_r = tfk.layers.Cropping2D(((0, 1), (0, 0)))(x_r)
            x_r = tfk.layers.ZeroPadding2D(((1, 0), (0, 0)))(x_r)

        x_l = tfk.layers.ZeroPadding2D(((0, 0), (self.n - 1, 0)))(x_l)
        x_r = tfk.layers.ZeroPadding2D(((0, 0), (0, self.n - 1)))(x_r)

        x_l = self.l_conv[0](x_l)
        x_r = self.r_conv[0](x_r)

        x_m = self.m_conv[0](x_m)
        x_m += x_l + x_r
        x_m = self.m_conv[1](x_m)
        if self.res:
            x_m2 = self.m_conv[2](x_m2)
            x_m = x_m + x_m2
        if not self.last_layer:
            x_l = self.l_conv[1](x_l)
            x_r = self.r_conv[1](x_r)

        return x_l, x_m, x_r


class BPnet(tfk.models.Model):
    def __init__(self, kernel_size, net_width, net_depth, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = kernel_size
        self.p = net_width
        self.d = net_depth
        self.lyr = [ConvBlock(self.p, self.n, 'A' if i == 0 else 'B', i == net_depth - 1)
                    for i in range(net_depth)]
        self.final_conv = tfk.layers.Conv2D(1, 1, activation=tfk.activations.sigmoid)
        self.final_conv.build([None, None, net_width])
        self.seed = tf.Variable(np.random.randint(-20000, 20000, 2, np.int32), False, dtype=tf.int32)
        # Above is the seed tensor for stateless random number generators
        self.learn_range = self.d * (self.n - 1)

    def call(self, x):
        x_l = x
        x_r = x
        x_m = tf.zeros_like(x)
        for i in range(self.d):
            x_l, x_m, x_r = self.lyr[i](x_l, x_m, x_r)
        return self.final_conv(x_m)

    def update_seed(self):
        self.seed.assign((self.seed * 1664525 + 1013904223) % 2 ** 31)
        # There's no actual problem with this

    @tf.function
    def _sample_update(self, x, pos):
        batch_size = x.shape[0]
        L = x.shape[1]
        r = self.learn_range
        full_zeros = tf.zeros([batch_size], tf.int32)
        full_ones = tf.ones([batch_size], tf.int32)
        i, j = tf.unstack(pos)
        self.update_seed()
        x = x[:, tfm.maximum(i - 1, 0):i + 1, tfm.maximum(j - r, 0):tfm.minimum(j + r + 1, L), :]
        x_hat = self.call(x)
        i_h = tfm.minimum(i, 1)
        j_h = tfm.minimum(j, r)

        if i == 0 and j == 0:
            probs = 0.5
        else:
            probs = x_hat[:, i_h, j_h, 0]
        indices = tf.stack([tf.range(batch_size), i * full_ones,
                            j * full_ones, full_zeros], 1)
        updates = tf.random.stateless_binomial([batch_size],
                                               self.seed, 1, probs, tf.float32) * 2 - 1
        return tf.tensor_scatter_nd_add(x, tf.cast(indices, tf.int32), updates)

    def sample(self, L, batch_size):
        x = tf.zeros([batch_size, L, L, 1])
        for i in range(L):
            for j in range(L):
                x = self._sample_update(x, tf.stack([i, j]))

        self.update_seed()
        flip = tf.random.stateless_binomial([batch_size, 1, 1, 1], self.seed, 1., 0.5,
                                            tf.float32) * 2 - 1
        x *= flip
        return x

    def _log_prob(self, x, x_hat):
        # Remember that x_hat gives prob of all 1's not given sample's
        mask = (x + 1) / 2
        log_prob = (
                tfm.log(x_hat + self.epsilon) * tf.cast(mask, tf.float32) +
                tfm.log(1 - x_hat + self.epsilon) * tf.cast(1 - mask, tf.float32))
        log_prob = tfm.reduce_sum(log_prob, [1, 2, 3])
        return log_prob

    def log_prob(self, x):
        x_hat = self.call(x)
        log_prob = self._log_prob(x, x_hat)
        if self.z2:
            x_inv = -x
            x_hat_inv = self.call(x_inv)
            log_prob_inv = self._log_prob(x_inv, x_hat_inv)
            log_prob = tfm.reduce_logsumexp(
                tf.stack([log_prob, log_prob_inv]),
                axis=0)
            log_prob -= tfm.log(2.)
        return tf.cast(log_prob, tf.float32)

import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow import math as tfm
import ising

class BoltzmannIsingBlock(tfk.layers.Layer):
    def __init__(self, out_features, kernel_size, mask_type, last_layer=False):
        super(BoltzmannIsingBlock, self).__init__()
        assert mask_type in ['A', 'B']
        self.p = out_features
        self.n = kernel_size
        self.last_layer = last_layer
        self.res = 0 if mask_type=='A' else 1

        k = self.res
        self.l_cropping = tfk.layers.Cropping2D(((0, 0), (0, 1-k)))
        self.l_padding = tfk.layers.ZeroPadding2D(((0, 0), (self.n-1, 0)))
        self.l_conv = tfk.layers.Conv2D(2*self.p, [1, self.n-1+k], activation='relu')
        self.l_conv2 = tfk.layers.Conv2D(self.p, 1, activation='tanh')

        self.r_cropping = tfk.layers.Cropping2D(((0, 1-k), (0, 0)))
        self.r_padding = tfk.layers.ZeroPadding2D(((1-k, 0), (0, self.n-1)))
        self.r_conv = tfk.layers.Conv2D(2*self.p, [1, self.n], activation='relu')
        self.r_conv2 = tfk.layers.Conv2D(self.p, 1, activation='tanh')

        self.m_conv = tfk.layers.Conv2D(2*self.p, 1)
        self.m_conv2 = tfk.layers.Conv2D(self.p, 1)
        if self.res:
            self.m_res_conv = tfk.layers.Conv2D(self.p, 1, use_bias=False)

    def call(self, x):
        if self.res:
            l_stack, m_stack, r_stack = tf.unstack(x, axis=-1)
            m_stack2 = m_stack
        else:
            l_stack = x
            r_stack = x
            m_stack = tf.zeros_like(x)
        #Left stack is cropped, padded followed by convolution
        l_stack = self.l_cropping(l_stack)
        l_stack = self.l_padding(l_stack)
        l_stack = self.l_conv(l_stack)

        #Right stack is cropped, padded followed by convolution
        r_stack = self.r_cropping(r_stack)
        r_stack = self.r_padding(r_stack)
        r_stack = self.r_conv(r_stack)

        #Update/initialise m_stack
        m_stack = self.m_conv(m_stack)
        m_stack += tfm.add(l_stack, r_stack)

        #Gating operation on m_stack
        m_stack0, m_stack1 = tf.split(m_stack, 2, axis=-1)
        m_stack0 = tfk.activations.tanh(m_stack0)
        m_stack1 = tfk.activations.sigmoid(m_stack1)
        m_stack = tfm.multiply(m_stack0, m_stack1)

        if not self.last_layer:
            l_stack = self.l_conv2(l_stack)
            r_stack = self.r_conv2(r_stack)

        #Convolve m_stack2, m_stack and connect them
        m_stack = self.m_conv2(m_stack)
        if self.res:
            m_stack2 = self.m_res_conv(m_stack2)
            m_stack = tfm.add(m_stack, m_stack2)

        if self.last_layer:
            output = m_stack
        else:
            output = tf.stack([l_stack, m_stack, r_stack], axis=-1)
        return output

class BPnet(ising.AutoregressiveModel):
    def __init__(self,  L, kernel_size, net_width, net_depth=None):
        super().__init__(L)
        if net_depth == None:
            assert type(net_width) == list
            net_depth = len(net_width)
            list_features = True
        else:
            list_features = False
        self.net_depth = net_depth
        self.net_width = net_width
        self.kernel_size = kernel_size
        layers = []
        out_features = self.net_width
        conv_block = BoltzmannIsingBlock
        if list_features:
            out_features = net_width[0]
        layers.append(conv_block(out_features, self.kernel_size, 'A',
                                 last_layer=True if self.net_depth == 1 else False))
        for i in range(self.net_depth-1):
            if list_features:
                out_features = net_width[i+1]
            layers.append(conv_block(
                out_features, self.kernel_size, 'B',
                last_layer=True if i == self.net_depth-2 else False))
        layers.append(tfk.layers.Conv2D(1, 1, activation='sigmoid'))
        self.custom_layers = layers
        #For use in sampling methods
        if list_features:
            self.learn_range = np.sum(net_width)-len(net_width)
        else:
            self.learn_range = net_depth*(net_width-1)

    def call(self, x):
        for i in range(self.net_depth+1):
            x = self.custom_layers[i](x)
        if self.z2 and x.shape[1] == self.L:
            x_hat = tfm.multiply(x, self.x_hat_mask)
            x_hat = tfm.add(x_hat, self.x_hat_bias)
        else:
            x_hat = x
        return x_hat

    def sample(self, batch_size):
        sample = np.zeros([batch_size, self.L, self.L, 1], np.float32)
        r = self.learn_range
        for i in range(self.L):
            for j in range(self.L):
                sub_sample = sample[:, np.maximum(
                    i-1, 0):i+1, np.maximum(j-r, 0):np.minimum(j+r+1, self.L)]
                x_hat = self.call(sub_sample)
                i_h = tfm.minimum(i, 1)
                j_h = tfm.minimum(j, r)
                probs = 0.5 if i == 0 and j == 0 else x_hat[:, i_h, j_h, :]
                sample[:, i, j, :] = np.random.binomial(1, probs,
                                                        [batch_size, 1])*2 - 1
        #x_hat = self.call(sample)
        if self.z2:
            flip = np.random.binomial(1, 0.5, [batch_size, 1, 1, 1])*2 - 1
            sample = sample*flip
        return sample

    def graph_sampler(self, batch_size, seed):
        #Same as sample method above but specialised for graph compilation
        sample = tf.zeros([batch_size, self.L, self.L, 1], tf.float32)
        tf_binomial = tf.random.stateless_binomial
        full_ones = tf.ones([batch_size], tf.int32)
        full_zeros = tf.zeros_like(full_ones)
        r = self.learn_range
        for i in range(self.L):
            for j in range(self.L):
                seed.assign((seed*1664525 + 1013904223) % 2**31)
                sub_sample = sample[:, np.maximum(i-1, 0):i+1,
                                    np.maximum(j-r, 0):np.minimum(j+r+1, self.L)]
                x_hat = self.call(sub_sample)
                i_h = tfm.minimum(i, 1)
                j_h = tfm.minimum(j, r)
                probs = 0.5 if i == 0 and j == 0 else x_hat[:, i_h, j_h, 0]
                indices = tf.stack(
                    [tf.range(batch_size), i*full_ones, j*full_ones, full_zeros], 1)
                updates = tf_binomial(
                    [batch_size], seed, 1., probs, tf.float32)*2 - 1
                sample = tf.tensor_scatter_nd_add(
                    sample, tf.cast(indices, tf.int32), updates)
        
        if self.z2:
            seed.assign((seed*1664525 + 1013904223) % 2**31)
            flip = tf_binomial([batch_size, 1, 1, 1], seed, 1., 0.5,
                               tf.float32)*2 - 1
            sample = sample*flip
        return sample

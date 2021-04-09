# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow import math as tfm
import ising
# %%
class MaskedConv2D(tfk.layers.Layer):
    def __init__(self, mask_type='A', filters=64, kernel_size=5, **kwargs):
        assert (kernel_size%2) == 1
        super(MaskedConv2D, self).__init__()
        self.conv = tfk.layers.Conv2D(filters, kernel_size, padding='same', **kwargs)
        assert mask_type in ['A','B']
        self.mask_type = mask_type
        self.mask = np.zeros([kernel_size, kernel_size, 1, filters])
        #Mask must be applied so that that the spin at a given position 
        #depends only on the spin values at positions that appear earlier in
        #the chosen autoregressive ordering
        self.mask[:kernel_size//2, ...] = 1
        self.mask[kernel_size//2, :kernel_size//2, ...] = 1
        if mask_type=='B':
            self.mask[kernel_size//2, kernel_size//2, ...] = 1

    def build(self, shape):
        self.conv.build(shape)

    def call(self, x):
        self.conv.kernel.assign(self.conv.kernel * self.mask)#type: ignore
        return self.conv(x)

class ResBlock(tfk.layers.Layer):
    def __init__(self, block):
        """Residual layer that adds a neural network block with its inputs

        Args:
            block (Keras Sequential block): Bijective block to add with its input
        """
        super(ResBlock, self).__init__()
        self.block = block

    def call(self, x):
        return x + self.block(x)
# %%
class PixelCNN(ising.AutoregressiveModel):
    def __init__(self, L=16, net_depth=3, net_width=64, kernel_size=5, res_block=True):
        super(PixelCNN, self).__init__(L, 0.0001)
        self.net_depth = net_depth
        self.net_width = net_width
        self.kernel_size = kernel_size
        self.res_block = res_block

        layers = []
        layers.append(tfk.layers.Input(shape=(self.L, self.L, 1)))
        layers.append(
            MaskedConv2D(
                mask_type='A',
                filters=1 if self.net_depth == 1 else self.net_width,
                kernel_size=self.kernel_size,
                activation='sigmoid' if self.net_depth == 1 else None
            ))
        layers.append(tfk.layers.LeakyReLU())
        layers.append(tfk.layers.Conv2D(self.net_width, 1))
        for _ in range(self.net_depth-2):
            if self.res_block:
                layers.append(
                    self._build_res_block())
            else:
                layers.append(
                    self._build_simple_block())
        if self.net_depth >= 2:
            layers.append(
                tfk.layers.Conv2D(1, 1, activation='sigmoid')
            )
        self.net = tfk.Sequential(layers)

    def _build_simple_block(self):
        layers = []
        layers.append(tfk.layers.LeakyReLU())
        layers.append(
            MaskedConv2D(
                mask_type='B',
                filters=self.net_width,
                kernel_size=self.kernel_size)
        )
        return tfk.Sequential(layers)

    def _build_res_block(self):
        layers = []
        layers.append(
            MaskedConv2D(
                mask_type='B',
                filters=self.net_width,
                kernel_size=self.kernel_size)
        )
        layers.append(tfk.layers.LeakyReLU())
        layers.append(tfk.layers.Conv2D(self.net_width, 1))
        return ResBlock(tfk.Sequential(layers))

#The following model and dependent layers an improvement over the
#PixelCNN model to counter blindspots in deep models in this class.
#GatedConvBlock has a multiplying gate which can improve the learning
#capability over PlainConvBlock, which only has convolutions with 
#residual connections
def AlphaConstraint(w):
    "Constraints w to range [0,1]"
    w = tfm.abs(w)
    return tfm.minimum(w, 1.0)

class PlainConvBlock(tfk.layers.Layer):
    def __init__(self, out_features, kernel_size, mask_type='A', last_layer=False):
        super(PlainConvBlock, self).__init__()
        self.p = out_features
        self.n = kernel_size
        self.last_layer = last_layer

        assert mask_type in ['A','B']
        k = 1 if mask_type == 'B' else 0
        #Definitions of variables/layers that initialize them must be 
        #made here, outside the call function
        self.hor_cropping = tfk.layers.Cropping2D(((0, 0), (0, 1-k)))
        self.hor_padding = tfk.layers.ZeroPadding2D(
            ((0, 0), (self.n//2, 0)))
        self.hor_conv = tfk.layers.Conv2D(self.p, [1, (self.n//2)+k])

        self.ver_cropping = tfk.layers.Cropping2D(((0, 1-k), (0, 0)))
        self.ver_padding = tfk.layers.ZeroPadding2D(((self.n//2, 0), 
                                                    (self.n//2, self.n//2)))
        self.ver_conv = tfk.layers.Conv2D(self.p, [(self.n//2)+k, self.n])
        self.res = k
        if self.res:
            self.res_conv = tfk.layers.Conv2D(self.p, 1, use_bias=False)

        self.sec_conv = tfk.layers.Conv2D(self.p, 1)
        self.alpha_scalar = tf.Variable(0.3, True, dtype=tf.float32, 
                                        constraint=AlphaConstraint)
        if not self.last_layer:
            self.sec_conv2 = tfk.layers.Conv2D(self.p, 1)
            self.alpha_scalar2 = tf.Variable(0.3, True, dtype=tf.float32,
                                             constraint=AlphaConstraint)
            
    def call(self, x):
        if self.res:
            h_stack, v_stack = tf.unstack(x, axis=-1)
        else:
            h_stack = x
            v_stack = x
        #Vertical stack is acted by a vertical convolution
        #equivalent to a masked one
        v_stack = self.ver_cropping(v_stack)
        v_stack = self.ver_padding(v_stack)
        v_stack = self.ver_conv(v_stack)

        #Horizontal stack is acted by a horizontal convolution
        #equivalent to a masked one- h_stack2 is kept for later
        h_stack2 = h_stack
        h_stack = self.hor_cropping(h_stack)
        h_stack = self.hor_padding(h_stack)
        h_stack = self.hor_conv(h_stack)

        #Connect horizontal and vertical stacks
        h_stack = tfm.add(h_stack, v_stack)

        #Convolve and act translationally invariant version of Prelu using
        #tf.maximum and passing user-defined scalar variable for alpha
        h_stack = tfm.maximum(self.alpha_scalar*h_stack, h_stack)
        h_stack = self.sec_conv(h_stack)
        if not self.last_layer:
            v_stack = tfm.maximum(self.alpha_scalar2*h_stack, h_stack)
            v_stack = self.sec_conv2(v_stack)

        #Make a residual connection between input state and output
        if self.res == 1:
            h_stack2 = self.res_conv(h_stack2)
            h_stack = tfm.add(h_stack, h_stack2)

        if self.last_layer:
            output = h_stack
        else:
            output = tf.stack([h_stack, v_stack], axis=-1)
        #Act with non-linear activation function
        return output

class GatedConvBlock(tfk.layers.Layer):
    def __init__(self, out_features, kernel_size=5, mask_type='A', last_layer=False):
        super(GatedConvBlock, self).__init__()
        assert out_features % 2 == 0
        self.p = out_features
        self.n = kernel_size
        self.last_layer = last_layer

        k = 1 if mask_type == 'B' else 0
        self.hor_cropping = tfk.layers.Cropping2D(((0, 0), (0, 1-k)))
        self.hor_padding = tfk.layers.ZeroPadding2D(
            ((0, 0), (self.n//2, 0)))
        self.hor_conv = tfk.layers.Conv2D(self.p, [1, (self.n//2)+k])

        self.ver_cropping = tfk.layers.Cropping2D(((0, 1-k), (0, 0)))
        self.ver_padding = tfk.layers.ZeroPadding2D(((self.n//2, 0),
                                                     (self.n//2, self.n//2)))
        self.ver_conv = tfk.layers.Conv2D(self.p, [(self.n//2)+k, self.n])

        self.ver_conv2 = tfk.layers.Conv2D(self.p//2, [1, 1])
        self.res = k
        self.hor_conv2 = tfk.layers.Conv2D(self.p//2, [1, 1])
        if self.res:
            self.res_conv = tfk.layers.Conv2D(self.p//2, [1, 1], use_bias=False)

    def call(self, x):
        if self.res == 1:
            h_stack, v_stack = tf.unstack(x, axis=-1)
        else:
            h_stack = x
            v_stack = x
        #Vertical stack is acted by a vertical convolution
        #equivalent to a masked one
        v_stack = self.ver_cropping(v_stack)
        v_stack = self.ver_padding(v_stack)
        v_stack = self.ver_conv(v_stack)

        #Horizontal stack is acted by a horizontal convolution
        #equivalent to a masked one- h_stack2 is kept for later
        h_stack2 = h_stack
        h_stack = self.hor_cropping(h_stack)
        h_stack = self.hor_padding(h_stack)
        h_stack = self.hor_conv(h_stack)

        #Add v_stack to h_stack
        h_stack = tfm.add(h_stack, v_stack)

        #"Gating" performed on horizontal stack
        h_stack0, h_stack1 = tf.split(h_stack, 2, axis=-1)
        h_stack0 = tfk.activations.tanh(h_stack0)
        h_stack1 = tfk.activations.sigmoid(h_stack1)
        h_stack = tfm.multiply(h_stack0, h_stack1)

        #"Gating" and convolving vertical stack
        if not self.last_layer:
            v_stack0, v_stack1 = tf.split(v_stack, 2, axis=-1)
            v_stack0 = tfk.activations.tanh(v_stack0)
            v_stack1 = tfk.activations.sigmoid(v_stack1)
            v_stack = tfm.multiply(v_stack0, v_stack1)

            v_stack = self.ver_conv2(v_stack)

        #Convolve h_stack2, h_stack and connect them
        h_stack = self.hor_conv2(h_stack)
        if self.res:
            h_stack2 = self.res_conv(h_stack2)
            h_stack = tfm.add(h_stack, h_stack2)

        if self.last_layer:
            output = h_stack
        else:
            output = tf.stack([h_stack, v_stack], axis=-1)
        return output

class NatConvBlock(GatedConvBlock):
    def __init__(self, out_features, kernel_size, mask_type, last_layer=False):
        super(NatConvBlock, self).__init__(out_features, kernel_size, mask_type, last_layer)
        self.p = out_features
        self.n = kernel_size
        self.last_layer = last_layer

        assert mask_type in ['A','B']
        k = 1 if mask_type == 'B' else 0
        self.hor_cropping = tfk.layers.Cropping2D(((0,0),(0,1-k)))
        self.hor_padding = tfk.layers.ZeroPadding2D(((0,0),(self.n-1,0)))
        self.hor_conv = tfk.layers.Conv2D(self.p, [1,self.n-1+k])

        self.ver_cropping = tfk.layers.Cropping2D(((0,1-k),(0,0)))
        self.ver_padding = tfk.layers.ZeroPadding2D(((1-k,0),(0,self.n-1)))
        self.ver_conv = tfk.layers.Conv2D(self.p, [1,self.n])

class AdvPixelCNN(ising.AutoregressiveModel):
    def __init__(self, L, kernel_size, net_width, net_depth=None, gated=False):
        super(AdvPixelCNN, self).__init__(L, 0.0001)
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
        layers.append(tfk.layers.Input((self.L, self.L, 1)))
        conv_block = NatConvBlock
        if list_features:
            out_features = net_width[0]
        layers.append(conv_block(out_features, self.kernel_size, 'A',
                                 last_layer=True if self.net_depth==1 else False))
        for i in range(self.net_depth-1):
            if list_features:
                out_features = net_width[i+1]
            layers.append(conv_block(
                out_features, self.kernel_size, 'B',
                last_layer=True if i==self.net_depth-2 else False))
        layers.append(tfk.layers.Conv2D(1, 1, activation='sigmoid'))

        self.net = tfk.Sequential(layers)

class VarConvBlock(tfk.layers.Layer):
    def __init__(self, out_features, kernel_size, mask_type, last_layer=False):
        super(VarConvBlock, self).__init__()
        self.p = out_features
        self.n = kernel_size
        self.last_layer = last_layer

        assert mask_type in ['A', 'B']
        k = 1 if mask_type == 'B' else 0

        self.hor_cropping = tfk.layers.Cropping2D(((0, 0), (0, 1-k)))
        self.hor_padding = tfk.layers.ZeroPadding2D(((0, 0), (self.n-1, 0)))
        self.hor_conv = tfk.layers.Conv2D(self.p, [1, self.n-1+k])
        self.hor_seq = tfk.Sequential([
            tfk.layers.Dense(self.p, "tanh"),
            tfk.layers.Dense(self.p, "tanh", use_bias=False)
        ])

        self.ver_cropping = tfk.layers.Cropping2D(((0, 1-k), (0, 0)))
        self.ver_padding = tfk.layers.ZeroPadding2D(((1-k, 0), (0, self.n-1)))
        self.ver_conv = tfk.layers.Conv2D(self.p, [1, self.n])
        self.ver_seq = tfk.Sequential([
            tfk.layers.Dense(self.p, "tanh"),
            tfk.layers.Dense(self.p, "tanh", use_bias=False)
        ])

        self.ver_conv2 = tfk.layers.Conv2D(self.p//2, [1, 1])
        self.res = k
        self.hor_conv2 = tfk.layers.Conv2D(self.p//2, [1, 1])
        if self.res:
            self.res_conv = tfk.layers.Conv2D(self.p//2,
                                              [1, 1], use_bias=False)

    def call(self, x, beta):
        if self.res == 1:
            h_stack, v_stack = tf.unstack(x, axis=-1)
        else:
            h_stack = x
            v_stack = x
        #Vertical stack is acted by a vertical convolution
        #equivalent to a masked one
        v_stack = self.ver_cropping(v_stack)
        v_stack = self.ver_padding(v_stack)
        v_stack = self.ver_conv(v_stack)
        v_stack += self.ver_seq(beta)

        #Horizontal stack is acted by a horizontal convolution
        #equivalent to a masked one- h_stack2 is kept for later
        h_stack2 = h_stack
        h_stack = self.hor_cropping(h_stack)
        h_stack = self.hor_padding(h_stack)
        h_stack = self.hor_conv(h_stack)
        h_stack = tfm.add(h_stack, self.hor_seq(beta))

        #Add v_stack to h_stack
        h_stack = tfm.add(h_stack, v_stack)

        #"Gating" performed on horizontal stack
        h_stack0, h_stack1 = tf.split(h_stack, 2, axis=-1)
        h_stack0 = tfk.activations.tanh(h_stack0)
        h_stack1 = tfk.activations.sigmoid(h_stack1)
        h_stack = tfm.multiply(h_stack0, h_stack1)

        #"Gating" and convolving vertical stack
        if not self.last_layer:
            v_stack0, v_stack1 = tf.split(v_stack, 2, axis=-1)
            v_stack0 = tfk.activations.tanh(v_stack0)
            v_stack1 = tfk.activations.sigmoid(v_stack1)
            v_stack = tfm.multiply(v_stack0, v_stack1)

            v_stack = self.ver_conv2(v_stack)

        #Convolve h_stack2, h_stack and connect them
        h_stack = self.hor_conv2(h_stack)
        if self.res:
            h_stack2 = self.res_conv(h_stack2)
            h_stack = tfm.add(h_stack, h_stack2)

        if self.last_layer:
            output = h_stack
        else:
            output = tf.stack([h_stack, v_stack], axis=-1)
        return output

class VarPixelCNN(ising.AutoregressiveModel):
    def __init__(self, L, kernel_size, net_width, net_depth=None, gated=False):
        super(VarPixelCNN, self).__init__(L, 0.0001)
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
        conv_block = VarConvBlock
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
        #Semantically, along the y axis, the given spin depends only on
        #"learn_range" spins before and after it

    def call(self, x, beta):
        for i in range(self.net_depth+1):
            if i==self.net_depth:
                x = self.custom_layers[i](x)
            else:
                x = self.custom_layers[i](x, beta)
        if self.z2 and x.shape[1]==self.L:
            x_hat = tfm.multiply(x, self.x_hat_mask)
            x_hat = tfm.add(x_hat, self.x_hat_bias)
        else:
            x_hat = x
        return x_hat

    def sample(self, batch_size, beta):
        beta = tf.convert_to_tensor(beta, tf.float32)
        sample = np.zeros([batch_size, self.L, self.L, 1], np.float32)
        r = self.learn_range
        for i in range(self.L):
            for j in range(self.L):
                sub_sample = sample[:, np.maximum(i-1, 0):i+1, np.maximum(j-r, 0):np.minimum(j+r+1,self.L)]
                x_hat = self.call(sub_sample, beta)
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

    def graph_sampler(self, batch_size, seed, beta):
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
                x_hat = self.call(sub_sample, beta)
                i_h = tfm.minimum(i, 1)
                j_h = tfm.minimum(j, r)
                probs = 0.5 if i == 0 and j == 0 else x_hat[:, i_h, j_h, 0]
                indices = tf.stack([tf.range(batch_size), i*full_ones, j*full_ones, full_zeros], 1)
                updates = tf_binomial([batch_size], seed, 1., probs, tf.float32)*2 - 1
                sample = tf.tensor_scatter_nd_add(sample, tf.cast(indices, tf.int32), updates)
                
        #x_hat = self.call(sample)
        if self.z2:
            seed.assign((seed*1664525 + 1013904223) % 2**31)
            flip = tf_binomial([batch_size, 1, 1, 1], seed, 1., 0.5,
                               tf.float32)*2 - 1
            sample = sample*flip
        return sample

    def log_prob(self, sample, beta):
        x_hat = self.call(sample, beta)
        log_prob = self._log_prob(sample, x_hat)
        if self.z2:
            sample_inv = -sample
            x_hat_inv = self.call(sample_inv, beta)
            log_prob_inv = self._log_prob(sample_inv, x_hat_inv)
            log_prob = tfm.reduce_logsumexp(
                tf.stack([log_prob, log_prob_inv]),
                axis=0)
            log_prob -= tfm.log(2.)
        return tf.cast(log_prob, tf.float32)
# %%

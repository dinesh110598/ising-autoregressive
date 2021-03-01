# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow import math as tfm
import ising
# %%
class MaskedConv2D(tfk.layers.Layer):
    def __init__(self, mask_type='A', filters=64, kernel_size=5, **kwargs):
        """Convolutional layer that's surjective and masks out kernel values in
        right half of the middle row and the entire bottom half. This is necessary
        in order to preserve the autoregressive property of every pixel

        Args:
            mask_type (str, optional): Determines whether the middle value of kernel
            is masked. Defaults to 'A'.
            filters (int, optional): No of output filters. Defaults to 64.
            kernel_size (int, optional): Size of convolution kernel. Defaults to 5.
            Other **kwargs pertaining to the regular conv2d layer may be passed
        """
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

class PixelCNN(tfk.Model):
    def __init__(self, L=10, net_depth=3, net_width=64, kernel_size=5, res_block=True, epsilon=0.001):
        """Builds a PixelCNN model that generates batches of Ising models of given length

        Args:
            L (int, optional): Length of Ising lattice. Defaults to 10.
            net_depth (int, optional): Depth of network. Defaults to 3.
            net_width (int, optional): No of output filters in all conv layers. Defaults to 64.
            kernel_size (int, optional): Size of kernel in all masked conv layers. Defaults to 5.
            res_block (bool, optional): Whether to have residual connections. Defaults to True.
            epsilon (float, optional): A very small quantity added to avoid log(0). 
                Defaults to 0.001.
        """
        super(PixelCNN, self).__init__()
        self.L = L
        self.net_depth = net_depth
        self.net_width = net_width
        self.kernel_size = kernel_size
        self.res_block = res_block
        self.epsilon = tf.cast(epsilon, tf.float32)

        self.x_hat_mask = np.ones([1, self.L, self.L, 1], np.float32)
        self.x_hat_mask[:,0,0,:] = 0
        self.x_hat_mask = tf.convert_to_tensor(self.x_hat_mask, tf.float32)
        self.x_hat_bias = np.zeros([1, self.L, self.L, 1], np.float32)
        self.x_hat_bias[:,0,0,:] = 0.5
        self.x_hat_bias = tf.convert_to_tensor(self.x_hat_bias, tf.float32)
        

        layers = []
        layers.append(tfk.layers.Input(shape= (self.L, self.L, 1)))
        layers.append(
            MaskedConv2D(
                mask_type='A',
                filters= 1 if self.net_depth==1 else self.net_width,
                kernel_size= self.kernel_size,
                activation='sigmoid' if self.net_depth==1 else None
            )
        )
        for _ in range(self.net_depth-2):
            if self.res_block:
                layers.append(
                    self._build_res_block())
            else:
                layers.append(
                    self._build_simple_block())
        if self.net_depth >= 2:
            layers.append(
                tfk.layers.LeakyReLU()
                )
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
                filters= self.net_width,
                kernel_size= self.kernel_size)
        )
        return tfk.Sequential(layers)
    
    def _build_res_block(self):
        layers = []
        layers.append(tfk.layers.Conv2D(self.net_width, 1))
        layers.append(tfk.layers.LeakyReLU())
        layers.append(
            MaskedConv2D(
                mask_type='B',
                filters= self.net_width,
                kernel_size= self.kernel_size)
        )
        return ResBlock(tfk.Sequential(layers))

    def call(self, x):
        x_hat = self.net(x)
        x_hat = x_hat*self.x_hat_mask + self.x_hat_bias#type:ignore
        return x_hat
    
    def sample(self, batch_size):
        sample = np.zeros([batch_size,self.L,self.L,1], np.float32)
        for i in range(self.L):
            for j in range(self.L):
                x_hat = self.call(sample).numpy()
                sample[:,i,j,:] = np.random.binomial(1, 
                                x_hat[:,i,j,:], [batch_size,1])*2 - 1
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
                                counts, x_hat[:,i,j,:], tf.float32)*2 - 1)
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
    

#Following constitute an alternative, improved autoregressive model
#with a design inspired from LSTM architecture for better learning capability
#They also have seperate horizontal and vertical stacks to counter the problem
#of blind spots in deep PixelCNN model
class GatedConvBlock(tfk.layers.Layer):
    def __init__(self, out_features, mask_type='A',kernel_size=5):
        super(GatedConvBlock, self).__init__()
        assert out_features%2 == 0
        self.p = out_features
        self.n = kernel_size
        
        if mask_type=='A':
            self.hor_cropping = tfk.layers.Cropping2D(((0,0),(0,1)))
            self.hor_padding = tfk.layers.ZeroPadding2D(((0,0),(self.n//2,0)))
            self.hor_conv = tfk.layers.Conv2D(self.p, [1,self.n//2])
            
            self.ver_cropping = tfk.layers.Cropping2D(((0,1),(0,0)))
            self.ver_padding = tfk.layers.ZeroPadding2D(((self.n//2,0),(self.n//2,self.n//2)))
            self.ver_conv = tfk.layers.Conv2D(self.p, [self.n//2, self.n])    
            
        elif mask_type=='B':
            self.hor_cropping = tfk.layers.Cropping2D()
            self.hor_padding = tfk.layers.ZeroPadding2D(((0,0),(self.n//2,0)))
            self.hor_conv = tfk.layers.Conv2D(self.p, [1,self.n//2+1])
                
            self.ver_cropping = tfk.layers.Cropping2D()
            self.ver_padding = tfk.layers.ZeroPadding2D(((self.n//2,0),(self.n//2,self.n//2)))
            self.ver_conv = tfk.layers.Conv2D(self.p, [(self.n//2)+1, self.n])
            
        
        
        self.res_conv = tfk.layers.Conv2D(self.p, [1,1], use_bias=False)
        self.res_conv2 = tfk.layers.Conv2D(self.p, [1,1])
        self.res_conv3 = tfk.layers.Conv2D(self.p, [1,1], use_bias=False)
    
    def call(self, x):
        h_stack, v_stack = tf.unstack(x, axis=-1)
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
        
        #Convolve v_stack and connect to h_stack
        v_stack2 = self.res_conv(v_stack)
        h_stack = tfm.add(h_stack, v_stack2)
        
        #"Gating" performed on horizontal stack        
        h_stack0, h_stack1 = tf.split(h_stack, 2, axis=-1)
        h_stack0 = tfk.activations.tanh(h_stack0)
        h_stack1 = tfk.activations.sigmoid(h_stack1)
        h_stack = tfm.multiply(h_stack0, h_stack1)
        
        #Convolve h_stack2, h_stack and connect them
        h_stack = self.res_conv2(h_stack)
        h_stack2 = self.res_conv3(h_stack2)
        h_stack = tfm.add(h_stack, h_stack2)
        
        return tf.stack([h_stack, v_stack], axis=-1)
    
class FinalConv(tfk.layers.Layer):
    def __init__(self):
        super(FinalConv, self).__init__()
        self.conv = tfk.layers.Conv2D(1, 1, activation='sigmoid')
    
    def call(self, x):
        x = tf.gather(x, 0, axis=-1)
        x = self.conv(x)
        return x

class GatedPixelCNN(tfk.Model):
    def __init__(self, L=24, kernel_size=3, net_depth=3, net_width=32,
                 epsilon=0.001):
        super(GatedPixelCNN, self).__init__()
        assert kernel_size%2 == 1
        assert net_depth > 0
        assert net_width > 0
        self.L = L
        self.net_depth = net_depth
        self.net_width = net_width
        self.kernel_size = kernel_size
        self.epsilon = tf.cast(epsilon, tf.float32)

        self.x_hat_mask = np.ones([1, self.L, self.L, 1], np.float32)
        self.x_hat_mask[:,0,0,:] = 0
        self.x_hat_bias = np.zeros([1, self.L, self.L, 1], np.float32)
        self.x_hat_bias[:,0,0,:] = 0.5
        
        layers = []
        layers.append( tfk.layers.Input((self.L, self.L, 1, 2)))
        layers.append( GatedConvBlock(self.net_width, 'A', self.kernel_size))
        
        for _ in range(self.net_depth-1):
            layers.append(tfk.layers.BatchNormalization())
            layers.append( GatedConvBlock(self.net_width, 'B', self.kernel_size))
        
        layers.append(FinalConv())
            
        self.net = tfk.Sequential(layers)
        
    def call(self, sample):
        x = tf.stack([sample, sample], axis=-1)
        x_hat = self.net(x)
        x_hat = tfm.multiply(x_hat,self.x_hat_mask)
        x_hat = tfm.add(x_hat, self.x_hat_bias)
        return x_hat
    
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
    
class GatedPCNN(ising.AutoregressiveModel):
    def __init__(self, L=24, kernel_size=3, net_depth=3, net_width=32,
                 epsilon=0.001):
        super(GatedPCNN, self).__init__(L, epsilon)
        assert kernel_size%2 == 1
        assert net_depth > 0
        assert net_width > 0
        self.net_depth = net_depth
        self.net_width = net_width
        self.kernel_size = kernel_size
        
        layers = []
        layers.append( tfk.layers.Input((self.L, self.L, 1, 2)))
        layers.append( GatedConvBlock(self.net_width, 'A', self.kernel_size))
        for _ in range(self.net_depth-1):
            layers.append( GatedConvBlock(self.net_width, 'B', self.kernel_size))
        layers.append(FinalConv())
            
        self.net = tfk.Sequential(layers)

    def call(self, sample):
        x = tf.stack([sample, sample], axis=-1)
        x_hat = self.net(x)
        x_hat = tfm.multiply(x_hat,self.x_hat_mask)
        x_hat = tfm.add(x_hat, self.x_hat_bias)
        return x_hat

# %%

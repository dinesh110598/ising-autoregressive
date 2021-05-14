import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow import math as tfm
import ising


class MaskedConv2D(tfk.layers.Layer):
    def __init__(self, mask_type='A', filters=64, kernel_size=5, **kwargs):
        assert (kernel_size % 2) == 1
        super(MaskedConv2D, self).__init__()
        self.conv = tfk.layers.Conv2D(
            filters, kernel_size, padding='same', **kwargs)
        assert mask_type in ['A', 'B']
        self.mask_type = mask_type
        self.mask = np.zeros([kernel_size, kernel_size, 1, filters])
        #Mask must be applied so that that the spin at a given position
        #depends only on the spin values at positions that appear earlier in
        #the chosen autoregressive ordering
        self.mask[:kernel_size//2, ...] = 1
        self.mask[kernel_size//2, :kernel_size//2, ...] = 1
        if mask_type == 'B':
            self.mask[kernel_size//2, kernel_size//2, ...] = 1

    def build(self, shape):
        self.conv.build(shape)

    def call(self, x):
        self.conv.kernel.assign(self.conv.kernel * self.mask)  # type: ignore
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

class SimplePCNN(ising.AutoregressiveModel):
    def __init__(self, L=16, net_depth=3, net_width=64, kernel_size=5, res_block=True):
        super(SimplePCNN, self).__init__(L, 0.0001)
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

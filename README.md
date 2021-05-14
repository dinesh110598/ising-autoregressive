# Autoregressive network on 2D Ising model
Aims to construct autoregressive networks (PixelCNN and similar) that outputs the log probabilities of configurations (and samples from it), given by Boltzmann distribution for Ising model to arbitrary accuracy. This uses Tensorflow 2 on a Jupyter notebook.

[![Open main.ipynb in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dinesh110598/ising-autoregressive/blob/main/main.ipynb)

In the graphs for Free energy predictions in various VarPCNN models, the variance is very small but the bias due to imperfect training seems to be larger. One very interesting observation is that the variance seems to reduce with increased lattice size, indicating a convergence to the thermodynamic limt.


"""
Author: Jacques LE THUAUT
Title: Implementation of a stochastic neural network class
File: arch_model.py
"""

from scipy import stats
from neural_network import NeuralNetwork
import numpy as np


class StochasticRecurrentNeuralNetwork(NeuralNetwork):

  def __init__(self, input_size, hidden_size, output_size):
    # Initialize network parameters randomly
    super().__init__(input_size + hidden_size, output_size) # Call the base class constructor with augmented input size
    self.U = np.random.randn(hidden_size, input_size) # Input-to-hidden weight matrix
    self.V = np.random.randn(hidden_size, hidden_size) # Hidden-to-hidden weight matrix
    self.c = np.random.randn(hidden_size) # Hidden bias vector
    self.h = np.zeros(hidden_size) # Hidden state vector
    self.z = np.zeros(output_size) # Latent variable vector

  def forward(self, x_seq):
    # Compute the output sequence and the latent sequence of the network given an input sequence x_seq
    y_seq = [] # Output sequence
    z_seq = [] # Latent sequence
    for x in x_seq: # For each input vector in the sequence
      self.h = np.tanh(np.dot(self.U, x) + np.dot(self.V, self.h) + self.c) # Update hidden state vector using tanh activation function
      self.z = np.dot(self.W, np.concatenate([x, self.h])) + self.b # Compute latent variable vector using linear transformation
      y = stats.norm.rvs(loc=self.z[0], scale=np.exp(self.z[1])) # Sample output vector from normal distribution parameterized by latent variable vector (mean and log variance)
      y_seq.append(y) # Append output vector to output sequence
      z_seq.append(self.z) # Append latent variable vector to latent sequence
    return y_seq, z_seq

  def backward(self, x_seq, y_seq, z_seq):
    # Compute the gradients of the loss function with respect to the network parameters given an input sequence x_seq, an output sequence y_seq, and a latent sequence z_seq 
    dU = np.zeros_like(self.U) # Gradient of input-to-hidden weight matrix
    dV = np.zeros_like(self.V) # Gradient of hidden-to-hidden weight matrix
    dc = np.zeros_like(self.c) # Gradient of hidden bias vector
    dW = np.zeros_like(self.W) # Gradient of weight matrix
    db = np.zeros_like(self.b) # Gradient of bias vector
    dh = np.zeros_like(self.h) # Gradient of hidden state vector
    dz = np.zeros_like(self.z) # Gradient of latent variable vector
    for x, y, z in reversed(list(zip(x_seq, y_seq, z_seq))): # For each input vector, output vector, and latent variable vector in the reversed sequence
      dy = (y - z[0]) / np.exp(z[1]) # Gradient of output vector
      dz[0] = dy - 1 # Gradient of mean latent variable
      dz[1] = -0.5 * (np.exp(z[1]) - 1 + dy**2) # Gradient of log variance latent variable
      dW_, db_, dxh = self.backward(np.concatenate([x, self.h]), z, dz) # Call the base class backward function with augmented input vector and latent variable vector as output vector
      dW += dW_ # Accumulate gradient of weight matrix
      db += db_ # Accumulate gradient of bias vector
      dx = dxh[:self.input_size] # Split gradient of augmented input vector into gradient of input vector
      dh += dxh[self.input_size:] # Add gradient of hidden state vector
      dh_ = (1 - self.h**2) * dh # Apply tanh derivative to gradient of hidden state vector
      dU += np.outer(dh_, x) # Accumulate gradient of input-to-hidden weight matrix
      dV += np.outer(dh_, self.h) # Accumulate gradient of hidden-to-hidden weight matrix
      dc += dh_ # Accumulate gradient of hidden bias vector
      dh = np.dot(self.V.T, dh_) # Update gradient of hidden state vector
    return dU, dV, dc, dW, db

  def update(self, dU, dV, dc, dW, db, lr):
    # Update the network parameters using stochastic gradient descent given gradients dU, dV, dc, dW and db and learning rate lr
    super().update(dW, db, lr) # Call the base class update function with gradients dW and db
    self.U -= lr * dU # Update input-to-hidden weight matrix
    self.V -= lr * dV # Update hidden-to-hidden weight matrix
    self.c -= lr * dc # Update hidden bias vector


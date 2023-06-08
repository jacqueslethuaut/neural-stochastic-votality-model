
"""
Author: Jacques LE THUAUT
Title: Implementation of a neural network class
File: arch_model.py
"""

import numpy as np
import scipy.stats as stats

import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, initialization='xavier'):
        self.initialization = initialization
        # Initialize weights and biases
        if self.initialization == 'xavier':
            self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
            self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        elif self.initialization == 'he':
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
        elif self.initialization == 'lecun':  # Added LeCun initialization
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1/input_size)
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1/hidden_size)
        
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward_pass(self, X):
        # Compute hidden layer
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.sigmoid(Z1)

        # Compute output layer
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.sigmoid(Z2)

        self.A1 = A1
        self.A2 = A2

        return A1, A2

    def backward_pass(self, X, y):
        m = X.shape[0]
        
        dZ2 = self.A2 - y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(self.A1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def update_weights(self, dW1, db1, dW2, db2, learning_rate):
        self.W1 = self.W1 - learning_rate * dW1
        self.b1 = self.b1 - learning_rate * db1
        self.W2 = self.W2 - learning_rate * dW2
        self.b2 = self.b2 - learning_rate * db2

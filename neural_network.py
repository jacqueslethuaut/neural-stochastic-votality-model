
"""
Author: Jacques LE THUAUT
Title: Implementation of a neural network class
File: arch_model.py
"""

import numpy as np

class NeuralNetwork:
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 initialization='xavier', 
                 lambda_=0.01, 
                 regularization='l2',
                 dropout_rate=0.5):
        
        self.initialization = initialization
        # regularization 
        self.lambda_ = lambda_  # regularization coefficient
        self.regularization = regularization
        # dropout
        self.dropout_rate = dropout_rate
        
        # Initialize weights and biases
        if self.initialization == 'xavier':
            self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
            self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        elif self.initialization == 'he':
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
        elif self.initialization == 'lecun':
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1/input_size)
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1/hidden_size)

        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward_pass(self, X, is_training=True):
        # Compute hidden layer
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.sigmoid(Z1)
        
        if is_training:
            # Dropout
            self.drop1 = (np.random.rand(*A1.shape) > self.dropout_rate)
            A1 *= self.drop1
        else:
            A1 *= (1 - self.dropout_rate)

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

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dZ1 *= self.drop1  # Apply dropout
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        # Apply regularization based on the selected type
        if self.regularization == 'l2':
            dW2 += (self.lambda_/m) * self.W2
            dW1 += (self.lambda_/m) * self.W1
        elif self.regularization == 'l1':
            dW2 += (self.lambda_/m) * np.sign(self.W2)
            dW1 += (self.lambda_/m) * np.sign(self.W1)
        elif self.regularization == 'elasticnet':
            dW2 += (self.lambda_/m) * np.sign(self.W2) + (self.lambda_/m) * self.W2
            dW1 += (self.lambda_/m) * np.sign(self.W1) + (self.lambda_/m) * self.W1

        return dW1, db1, dW2, db2

    def update_weights(self, dW1, db1, dW2, db2, learning_rate):
        self.W1 = self.W1 - learning_rate * dW1
        self.b1 = self.b1 - learning_rate * db1
        self.W2 = self.W2 - learning_rate * dW2
        self.b2 = self.b2 - learning_rate * db2
        
    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = (-1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

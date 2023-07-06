"""
Author: Jacques LE THUAUT

Title: Implementation of a neural network class
"""

import numpy as np

class NeuralNetwork:
    """
    Shows the NeuralNetwork class and its evolution.
    
    Usage:

    .. math::

        dZ^{[L]} = A^{[L]} - Y
    """
    
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 initialization='xavier', 
                 lambda_=0.01, 
                 regularization='l2', 
                 dropout_rate=0.5):
        """
        This function initializes weights and biases for a neural network with options for different initialization methods, regularization, and dropout.
        
        :param input_size: The number of input features in the input data
        :param hidden_size: The number of neurons in the hidden layer of a neural network
        :param output_size: The number of output neurons in the neural network. It determines the dimensionality of the output of the network
        :param initialization: The method used to initialize the weights of the neural network. It can be 'xavier', 'he', or 'lecun', defaults to xavier (optional)
        :param lambda_: The regularization coefficient used in the regularization term of the loss function. It controls the strength of the regularization and helps prevent overfitting
        :param regularization: Regularization is a technique used to prevent overfitting in machine learning models. It involves adding a penalty term to the loss function during training, which encourages the model to learn simpler patterns that generalize better to new data. The regularization coefficient (lambda_) controls the strength of the penalty term, and the, defaults to l2 (optional)
        :param dropout_rate: dropout_rate is a hyperparametereter that controls the probability of dropping out a neuron during training in order to prevent overfitting. It is the probability of setting a neuron's output to zero during training. A dropout rate of 0.5 means that each neuron has a 50% chance of being activated
        """
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
        """
        The sigmoid function returns the output of the logistic function for a given input.
        
        :param z: z is the input to the sigmoid function. It can be a scalar or a numpy array of any shape.The sigmoid function returns a value between 0 and 1, which represents the probability of a binary outcome
        
        :return: The function `sigmoid` is returning the output of the sigmoid function applied to the input `z`. The sigmoid function is defined as `1 / (1 + exp(-z))`, where `exp` is the exponential function and `z` is the input.
        
        .. math:: 
            1 / (1 + exp(-z))
        
        """
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        """
        The function calculates the derivative of the sigmoid function with respect to its input.
        
        :param z: The parameter "z" is the input to the sigmoid function. It can be a scalar value or a vector/matrix of values. The sigmoid function takes this input and returns a value between 0 and 1, which represents the probability of a certain event occurring. 
        
        :return: the derivative of the sigmoid function evaluated at the input value z. The derivative is calculated as the sigmoid of z multiplied by 1 minus the sigmoid of z.
        
        .. math:: 
            sigmoid(z) * (1 - self.sigmoid(z))
        """
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward_pass(self, X, is_training=True):
        """
        This function performs a forward pass through a neural network with a hidden layer and dropout regularization.
        
        :param X: The input data to the neural network. It is a matrix of shape (batch_size, input_size), where batch_size is the number of examples in a batch and input_size is the number of features in each example
        :param is_training: The "is_training" parameter is a boolean variable that indicates whether the forward pass is being performed during training or testing. If it is set to True, then dropout regularization is applied to the hidden layer activations to prevent overfitting. If it is set to False, then dropout is not applied and, defaults to True (optional)
        
        :return: The forward_pass method returns the activations of the hidden layer (A1) and the output layer (A2).
        """
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
        """
        This function computes the gradients of the weights and biases for a neural network during back propagation, with optional regularization.
        
        :param X: The input data matrix of shape (m, n), where m is the number of examples and n is the number of features
        :param y: y is the true label or target output of the input data X. It is used in the backward pass to calculate the error between the predicted output and the true output
        
        :return: the gradients of the weights and biases of the neural network, which are dW1, db1, dW2, and db2. These gradients are calculated using the back propagation algorithm and are used to update the weights and biases during the training process.
        """
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
        """
        This function updates the weights and biases of a neural network using the given gradients and
        learning rate.
        
        :param dW1: The gradient of the loss function with respect to the weights of the first layer
        :param db1: db1 is the gradient of the loss function with respect to the bias of the first layer in a neural network. It is used in the backpropagation algorithm to update the bias of the first layer during training
        :param dW2: dW2 is the gradient of the loss function with respect to the weights of the second layer in a neural network. It is calculated during backpropagation and represents the direction and magnitude of the change needed to the weights in order to reduce the loss. The update_weights function uses this gradient, along
        :param db2: db2 is the gradient of the loss function with respect to the bias of the output layer. It is used to update the bias of the output layer during back propagation in order to minimize the loss function
        :param learning_rate: The learning rate is a hyperparameter that determines the step size at which the model parameters (weights and biases) are updated during training. It controls how much the model adjusts its parameters in response to the estimated error each time the model weights are updated. A higher learning rate can result in faster convergence during
        
        """
        self.W1 = self.W1 - learning_rate * dW1
        self.b1 = self.b1 - learning_rate * db1
        self.W2 = self.W2 - learning_rate * dW2
        self.b2 = self.b2 - learning_rate * db2
        
    def compute_loss(self, y_true, y_pred):
        """
        This function computes the binary cross-entropy loss between the true labels and predicted probabilities.
        
        :param y_true: The true labels or target values of the data
        :param y_pred: y_pred is the predicted output of the model for a given input. In other words, it is the output of the last layer of the neural network after passing the input through the network. It is a vector of probabilities that represents the likelihood of each class being the correct classification for the input
        
        :return: the loss value calculated using the binary cross-entropy loss formula.
        
        """
        m = y_true.shape[0]
        return (-1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

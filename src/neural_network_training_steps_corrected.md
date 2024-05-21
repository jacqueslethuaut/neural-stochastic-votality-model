
# Neural Network Training Steps  

## Step 1: Initialization  

Initialize weights $W^{l}$ and biases $b^{l}$ for each layer $l$ with random values.

## Step 2: Forward Propagation  

For a given input $X$ and each layer $l$ from 1 to $L$ (the output layer):
$$
Z^{l} = W^{l} A^{l-1} + b^{l}
$$  
$$
A^{l} = f^{l}(Z^{l})
$$
where $A^{0} = X$ is the input, $Z^{l}$ is the pre-activation, $A^{l}$ is the activation using the activation function $f^{l}$, and $L$ is the total number of layers.

## Step 3: Loss Calculation
Compute the loss function $L$ which is a function of the predicted output $y_{pred} = A^{L}$ and the true output $y_{true}$:
$$
L = L(y_{pred}, y_{true})
$$
For Mean Squared Error, this would be:
$$
L = rac{1}{2} \sum_{i}(y_{pred}^{i} - y_{true}^{i})^2
$$

## Step 4: Backpropagation
Calculate the gradient of the loss $L$ with respect to the weights $W^{l}$ and biases $b^{l}$ for each layer starting from the output layer and moving backwards.

For the output layer $L$, the gradient of the loss with respect to weights is:
$$
rac{\partial L}{\partial W^{L}} = rac{\partial L}{\partial A^{L}} \cdot rac{\partial A^{L}}{\partial Z^{L}} \cdot rac{\partial Z^{L}}{\partial W^{L}}
$$

For the hidden layers $l = L-1, L-2, ..., 2, 1$, the gradients are calculated recursively:
$$
rac{\partial L}{\partial W^{l}} = rac{\partial L}{\partial Z^{l+1}} \cdot rac{\partial Z^{l+1}}{\partial A^{l}} \cdot rac{\partial A^{l}}{\partial Z^{l}} \cdot rac{\partial Z^{l}}{\partial W^{l}}
$$

## Step 5: Gradient Descent Update
Update the weights and biases using gradient descent:
$$
W^{l}_{	ext{new}} = W^{l} - \eta rac{\partial L}{\partial W^{l}}
$$
$$
b^{l}_{	ext{new}} = b^{l} - \eta rac{\partial L}{\partial b^{l}}
$$
where $\eta$ is the learning rate.

## Step 6: Iteration
Repeat Steps 2 to 5 for a set number of epochs or until early stopping criteria are met.

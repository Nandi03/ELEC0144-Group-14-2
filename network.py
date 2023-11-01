import numpy as np
import matplotlib.pyplot as plt
from tsensor import explain as exp

# Define the derivative of tanh function
def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

# Find the mean squared error
def mse(actual, predicted):
    return 0.5*(actual - predicted) ** 2

# derivative of mean squared error to find the direction to go in
def mse_grad(actual, predicted):
    return (actual - predicted)

# Initialise the weights randomly
np.random.seed(42)
l1_weights = np.array([[0.1, 0.1, 0.1]])
l2_weights = np.array([[0.1], [0.1], [0.1]])
# Learning rate
learning_rate = 0.00001

# Number of epochs
epochs = 1000

# Training data
x_train = np.arange(-1, 1, 0.05)
d_train = 0.8* x_train**3 + 0.3 * x_train**2 - 0.4*x_train + np.random.normal(0, 0.02, len(x_train))


# Training the neural network using stochastic gradient descent
for epoch in range(epochs):
    # Iterate through each training example
    for i in range(len(x_train)):
        # Forward pass
        l1_output =  np.dot(x_train[i], l1_weights) # Layer 1
        l1_activated = np.tanh(l1_output) # Apply tanh activation func
        output = np.dot(l1_activated, l2_weights) # Layer 2

        # Back-propagation
        output_grad = mse_grad(d_train[i], output)
        l2_w_grad = np.dot(l1_activated.T, output_grad)

        l1_activated_grad = np.dot(output_grad, l2_weights.T)
        l1_output_grad = tanh_derivative(l1_activated_grad)

        l1_w_grad = np.dot(x_train[i], l1_output_grad)

        l2_weights = np.subtract(l2_weights, l2_w_grad * learning_rate)
        l1_weights = np.subtract(l1_weights, l1_w_grad * learning_rate)

y = []

for i in range(len(x_train)):
    l1_output =  np.dot(x_train[i], l1_weights) # Layer 1
    l1_activated = np.tanh(l1_output) # Apply tanh activation func
    output = np.sum(np.dot(l1_activated, l2_weights)) # Layer 2

    y.append(output)

# Plot the input-output relationship
plt.figure()
plt.scatter(x_train, d_train, label='Training Data')  # Training data points
plt.plot(x_train, y, color='red', label='Predicted Output')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('1-3-1 Neural Network with Tanh Activation')
plt.legend()
plt.show()

print(l1_weights)

        


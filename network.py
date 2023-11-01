import numpy as np
import matplotlib.pyplot as plt
from tsensor import explain as exp

def mse(actual, predicted):
    return (actual - predicted)**2

def mse_grad(actual, predicted):
    return predicted - actual

# Define the derivative of tanh function
def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

# Initialise the weights randomly
np.random.seed(42)
l1_weights = np.random.randn(3, 1)
l2_weights = np.random.randn(1, 1)

# Learning rate
learning_rate = 0.0005

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
        # This is the output of the layer
      

        l1_output = np.dot(x_train[i],l1_weights)
        l1_activated = np.array(np.tanh(l1_output))
        l2_output = np.array(np.sum(np.dot( l1_activated, l2_weights ))).reshape(1, 1)

        # Compute the error
        error = mse(d_train[i], l2_output)

        # Backpropagation
        error_grad = mse_grad(d_train[i],l2_output)
        output_grad = mse_grad(d_train[i], l2_output)
        
        # Update weights using gradient descent
        l2_w_gradient =  np.dot(l1_activated, output_grad) 
        diff = np.dot(l2_w_gradient, learning_rate)
        l1_weights = np.subtract(l1_weights, diff)

        l1_activated_grad = np.dot(output_grad,l2_weights)
        l1_output_gradient = np.dot( (np.subtract(np.array([[1.0], [1.0], [1.0]]), np.exp(l1_output, np.array([[2.0], [2.0], [2.0]])))), l1_activated_grad )
        l1_w_gradient = np.dot(x_train[i] , l1_output_gradient)

        l1_weights -= np.dot(l1_w_gradient , learning_rate)

        

y = []

for i in range(len(x_train)):
    l1_output = np.dot(x_train[i],l1_weights)
    l1_activated = np.array(np.tanh(l1_output))
    l2_output = (np.sum(np.dot( l1_activated, l2_weights )))

    y.append(l2_output)

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
print(l2_weights)



import numpy as np
import matplotlib.pyplot as plt
from tsensor import explain as exp

# Activation functions and their derivatives
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

def mse(actual, predicted):
    return 0.5 * ((predicted- actual)**2)

def mse_grad(actual, predicted):
    return predicted - actual

# Initialize weights randomly
np.random.seed(42)
input_size = 1
hidden_size = 3
output_size = 1

l1_weights = np.random.randn(input_size, hidden_size) * np.sqrt(2 / (input_size + hidden_size))
l1_bias = np.zeros((1, hidden_size))  # Bias for hidden layer
l2_weights = np.random.randn(hidden_size, output_size) * np.sqrt(2 / (hidden_size + output_size))
l2_bias = np.zeros((1, output_size))  # Bias for output layer

# Layer 1 = sigmoid
# Layer 2 =  linear

# Learning rate
learning_rate = 0.001

# Number of epochs
epochs = 1000

# Training data
x_train = np.arange(-1, 1, 0.05)
d_train = 0.8* x_train**3 + 0.3 * x_train**2 - 0.4*x_train + np.random.normal(0, 0.02, len(x_train))


# Training the neural network using stochastic gradient descent
for epoch in range(epochs):
    # Iterate through each training example
    epoch_loss = 0
    for i in range(len(x_train)):
        # Forward pass
        l1_output =  np.dot(x_train[i], l1_weights) + l1_bias # Layer 1
        l1_activated = 1 / (1 + np.exp(-l1_output)) # Apply sigmoid activation func
        l2_output = np.dot(l1_activated, l2_weights) + l2_bias # Layer 2
        l2_activated = np.tanh(l2_output)

        # Compute loss
        loss = mse(d_train[i], l2_activated)

        # Back-propagation
        output_grad = mse_grad(d_train[i], l2_activated) * tanh_derivative(l2_output)
        l1_output_grad = np.dot(output_grad, l2_weights.T) * tanh_derivative(l1_output)
        # update weights using gradient descent
        l1_weights -= learning_rate * np.outer(x_train[i], l1_output_grad)
        l1_bias -= learning_rate * l1_output_grad
        
        l2_weights -= learning_rate * np.outer(l1_activated, output_grad)
        l2_bias -= learning_rate * output_grad

        # Compute individual sample loss
        sample_loss = 0.5 * ((l2_activated - d_train[i])**2)

        # Accumulate sample loss to calculate mean loss for the epoch
        epoch_loss += sample_loss
     # Print loss every 100 epochs
    mean_epoch_loss = np.mean(epoch_loss)

    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Mean Loss: {mean_epoch_loss:.6f}')


x_test = np.arange(-0.97, 0.93, 0.1)
y_test = 0.8* x_test**3 + 0.3 * x_test**2 - 0.4*x_test + np.random.normal(0, 0.02, len(x_test))
predictions = []
for i in range(len(x_test)):
    l1_output =  np.dot(x_train[i], l1_weights) + l1_bias # Layer 1
    l1_activated = 1 / (1 + np.exp(-l1_output)) # Apply sigmoid activation func
    l2_output = np.dot(l1_activated, l2_weights) + l2_bias # Layer 2
    l2_activated = np.sum(np.tanh(l2_output))
    predictions.append(l2_activated)

# Plot the training data, true cubic function, and predictions
plt.figure(figsize=(8, 6))
plt.scatter(x_test, y_test, color='blue', label='Training Data')
plt.plot(x_test, predictions, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=8, label='Predictions')
plt.xlabel('X_test')
plt.ylabel('Y')
plt.title('Cubic Function Approximation using Neural Network (Sequential Learning)')
plt.legend()
plt.grid(True)
plt.show()




#---------------------------------------------------------------------------------------------
# Layer 1 = tanh
# Layer 2 = tanh
        
import numpy as np
import matplotlib.pyplot as plt

# Generate some random training data
X_train = np.arange(-1, 1, 0.05).reshape(-1, 1)
Y_train = 0.8 * X_train**3 + 0.3 * X_train**2 - 0.4 * X_train + np.random.normal(0, 0.02, len(X_train)).reshape(-1, 1)

# Activation functions and their derivatives
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

# Initialize weights and biases randomly
input_size = 1
hidden_size = 3
output_size = 1

np.random.seed(42)
weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2 / (input_size + hidden_size))
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2 / (hidden_size + output_size))
bias_output = np.zeros((1, output_size))

# Learning rate
learning_rate = 0.01

# Number of epochs
epochs = 1000

# Training the neural network using stochastic gradient descent
for epoch in range(epochs):
    hidden_input = np.dot(X_train, weights_input_hidden) + bias_hidden
    hidden_output = tanh(hidden_input)
    output = np.dot(hidden_output, weights_hidden_output) + bias_output
    
    # Apply Tanh activation function to Layer 2 output
    output = tanh(output)
    
    # Compute loss
    loss = 0.5 * np.mean((output - Y_train)**2)
    
    # Back-propagation
    output_grad = (output - Y_train) * tanh_derivative(output)
    hidden_output_grad = np.dot(output_grad, weights_hidden_output.T) * tanh_derivative(hidden_input)
    
    # Update weights and biases using gradient descent
    weights_hidden_output -= learning_rate * np.dot(hidden_output.T, output_grad)
    bias_output -= learning_rate * np.sum(output_grad, axis=0, keepdims=True)
    weights_input_hidden -= learning_rate * np.dot(X_train.T, hidden_output_grad)
    bias_hidden -= learning_rate * np.sum(hidden_output_grad, axis=0, keepdims=True)
    
    # Print mean loss for the epoch every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Mean Loss: {loss:.6f}')

# Test the model with some sample data
X_test = np.arange(-1, 1, 0.05).reshape(-1, 1)
hidden_input_test = np.dot(X_test, weights_input_hidden) + bias_hidden
hidden_output_test = tanh(hidden_input_test)
predictions = np.dot(hidden_output_test, weights_hidden_output) + bias_output

# Plot the training data, true cubic function, and predictions
plt.figure(figsize=(8, 6))
plt.scatter(X_train, Y_train, color='blue', label='Training Data')
plt.plot(X_test, predictions, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=8, label='Predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Cubic Function Approximation using Neural Network with Tanh Activation')
plt.legend()
plt.grid(True)
plt.show()
# ---------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Generate some random training data
X_train = np.arange(-1, 1, 0.05).reshape(-1, 1)
Y_train = 0.8 * X_train**3 + 0.3 * X_train**2 - 0.4 * X_train + np.random.normal(0, 0.02, len(X_train)).reshape(-1, 1)

# Activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Initialize weights and biases randomly
input_size = 1
hidden_size = 3
output_size = 1

np.random.seed(42)
weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2 / (input_size + hidden_size))
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2 / (hidden_size + output_size))
bias_output = np.zeros((1, output_size))

# Learning rate
learning_rate = 0.01

# Number of epochs
epochs = 1000

# Training the neural network using stochastic gradient descent
for epoch in range(epochs):
    hidden_input = np.dot(X_train, weights_input_hidden) + bias_hidden
    hidden_output = relu(hidden_input)
    output = np.dot(hidden_output, weights_hidden_output) + bias_output
    
    # Compute loss
    loss = 0.5 * np.mean((output - Y_train)**2)
    
    # Back-propagation
    output_grad = (output - Y_train) * relu_derivative(output)
    hidden_output_grad = np.dot(output_grad, weights_hidden_output.T) * relu_derivative(hidden_input)
    
    # Update weights and biases using gradient descent
    weights_hidden_output -= learning_rate * np.dot(hidden_output.T, output_grad)
    bias_output -= learning_rate * np.sum(output_grad, axis=0, keepdims=True)
    weights_input_hidden -= learning_rate * np.dot(X_train.T, hidden_output_grad)
    bias_hidden -= learning_rate * np.sum(hidden_output_grad, axis=0, keepdims=True)
    
    # Print mean loss for the epoch every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Mean Loss: {loss:.6f}')

# Test the model with some sample data
X_test = np.arange(-1, 1, 0.05).reshape(-1, 1)
hidden_input_test = np.dot(X_test, weights_input_hidden) + bias_hidden
hidden_output_test = relu(hidden_input_test)
predictions = np.dot(hidden_output_test, weights_hidden_output) + bias_output

# Plot the training data, true cubic function, and predictions
plt.figure(figsize=(8, 6))
plt.scatter(X_train, Y_train, color='blue', label='Training Data')
plt.plot(X_test, predictions, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=8, label='Predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Cubic Function Approximation using Neural Network with ReLU Activation')
plt.legend()
plt.grid(True)
plt.show()

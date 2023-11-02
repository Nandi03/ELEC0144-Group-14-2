import numpy as np
import matplotlib.pyplot as plt

# Generate some random training data
x_train = np.arange(-1, 1, 0.05)
d_train = 0.8 * x_train**3 + 0.3 * x_train**2 - 0.4 * x_train + np.random.normal(0, 0.02, len(x_train))

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


# Learning rate
learning_rate = 0.1

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
        l1_activated = np.tanh(l1_output) # Apply tanh activation func
        output = np.dot(l1_activated, l2_weights) + l2_bias # Layer 2

        # Compute loss
        loss = mse(d_train[i], output)

        # Back-propagation
        output_grad = mse_grad(d_train[i], output)
        l1_output_grad = np.dot(output_grad, l2_weights.T) * tanh_derivative(l1_output)

        # update weights using gradient descent
        l1_weights -= learning_rate * np.outer(x_train[i], l1_output_grad)
        l1_bias -= learning_rate * l1_output_grad
        
        l2_weights -= learning_rate * np.outer(l1_activated, output_grad)
        l2_bias -= learning_rate * output_grad

        # Compute individual sample loss
        sample_loss = 0.5 * ((output - d_train[i])**2)

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
    l1_input = np.dot(x_test[i], l1_weights) + l1_bias
    l1_activated = np.tanh(l1_input)
    output = np.sum(np.dot(l1_activated, l2_weights) + l2_bias)
    predictions.append(output)

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

        


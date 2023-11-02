import numpy as np
import matplotlib.pyplot as plt

# Generate some random training data
X_train = np.arange(-1, 1, 0.05)
Y_train = 0.8 * X_train**3 + 0.3 * X_train**2 - 0.4 * X_train + np.random.normal(0, 0.02, len(X_train))

# Activation functions and their derivatives
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

# Initialize weights randomly
np.random.seed(42)
input_size = 1
hidden_size = 3
output_size = 1

weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2 / (input_size + hidden_size))
bias_hidden = np.zeros((1, hidden_size))  # Bias for hidden layer
weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2 / (hidden_size + output_size))
bias_output = np.zeros((1, output_size))  # Bias for output layer


# Learning rate
learning_rate = 0.1

# Number of epochs
epochs = 1000

# Training the neural network using sequential updates
for epoch in range(epochs):
    epoch_loss = 0
    for i in range(len(X_train)):
        # Forward pass
        hidden_input = np.dot(X_train[i], weights_input_hidden) + bias_hidden
        hidden_output = tanh(hidden_input)
        output = np.dot(hidden_output, weights_hidden_output) + bias_output
        
        # Compute loss
        loss = 0.5 * ((output - Y_train[i])**2)
        
        # Backpropagation
        output_error = output - Y_train[i]
        hidden_error = np.dot(output_error, weights_hidden_output.T) * tanh_derivative(hidden_input)
        
        # Update weights using gradient descent
        weights_hidden_output -= learning_rate * np.outer(hidden_output, output_error)
        bias_output -= learning_rate * output_error

        weights_input_hidden -= learning_rate * np.outer(X_train[i], hidden_error)
        bias_hidden -= learning_rate * hidden_error

         # Compute individual sample loss
        sample_loss = 0.5 * ((output - Y_train[i])**2)
        
        # Accumulate sample loss to calculate mean loss for the epoch
        epoch_loss += sample_loss
    # Print loss every 100 epochs
    mean_epoch_loss = np.mean(epoch_loss)

    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Mean Loss: {mean_epoch_loss:.6f}')

# Test the model with some sample data
X_test = np.arange(-1, 1, 0.05)
predictions = []
for i in range(len(X_test)):
    hidden_input = np.dot(X_test[i], weights_input_hidden) + bias_hidden
    hidden_output = tanh(hidden_input)
    output = np.sum(np.dot(hidden_output, weights_hidden_output) + bias_output)
    predictions.append(output)

# Plot the training data, true cubic function, and predictions
plt.figure(figsize=(8, 6))
plt.scatter(X_train, Y_train, color='blue', label='Training Data')
plt.plot(X_test, predictions, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=8, label='Predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Cubic Function Approximation using Neural Network (Sequential Learning)')
plt.legend()
plt.grid(True)
plt.show()

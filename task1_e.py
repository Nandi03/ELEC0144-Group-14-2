import numpy as np
import matplotlib.pyplot as plt

# Activation functions and their derivatives
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

def mse(actual, predicted):
    return 0.5 * ((predicted- actual)**2)

def mse_grad(actual, predicted):
    return predicted - actual

# Define the training data
x_train = np.arange(-1, 1, 0.05)
d_train = 0.8 * x_train**3 + 0.3 * x_train**2 - 0.4 * x_train + np.random.normal(0, 0.02, len(x_train))

# Initialize network parameters
np.random.seed(42)
input_size = 1
hidden_size = 3
output_size = 1

l1_weights = np.random.randn(input_size, hidden_size)
l1_bias = np.random.randn(1, hidden_size)
l2_weights = np.random.randn(hidden_size, output_size)
l2_bias  = np.random.randn(1, output_size)

# Hyperparameters
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# ADAM variables
m_W1, m_B1, m_W2, m_B2 = 0, 0, 0, 0
v_W1, v_B1, v_W2, v_B2 = 0, 0, 0, 0
t = 0

# Training loop
epochs = 1000

for epoch in range(epochs):
    for i in range(len(x_train)):
        # Forward propagation
        X = x_train[i]
        Z1 = np.dot(X, l1_weights) + l1_bias
        A1 = tanh(Z1)
        Z2 = np.dot(A1, l2_weights) + l2_bias
        Y = Z2

        # Compute loss
        loss = mse(d_train[i], Y)

        # Backpropagation
        delta2 = mse_grad(d_train[i], Y)
        dW2 = np.dot(A1.T, delta2)
        dB2 = delta2
        delta1 = np.dot(delta2, l2_weights.T) * tanh_derivative(A1)
        dW1 = np.dot(X.T, delta1)
        dB1 = delta1

        # Update ADAM variables
        t += 1
        m_W1 = beta1 * m_W1 + (1 - beta1) * dW1
        m_B1 = beta1 * m_B1 + (1 - beta1) * dB1
        m_W2 = beta1 * m_W2 + (1 - beta1) * dW2
        m_B2 = beta1 * m_B2 + (1 - beta1) * dB2

        v_W1 = beta2 * v_W1 + (1 - beta2) * (dW1**2)
        v_B1 = beta2 * v_B1 + (1 - beta2) * (dB1**2)
        v_W2 = beta2 * v_W2 + (1 - beta2) * (dW2**2)
        v_B2 = beta2 * v_B2 + (1 - beta2) * (dB2**2)

        # Bias correction
        m_W1_hat = m_W1 / (1 - beta1**t)
        m_B1_hat = m_B1 / (1 - beta1**t)
        m_W2_hat = m_W2 / (1 - beta1**t)
        m_B2_hat = m_B2 / (1 - beta1**t)
        
        v_W1_hat = v_W1 / (1 - beta2**t)
        v_B1_hat = v_B1 / (1 - beta2**t)
        v_W2_hat = v_W2 / (1 - beta2**t)
        v_B2_hat = v_B2 / (1 - beta2**t)

        # Update weights and biases
        l1_weights -= learning_rate * m_W1_hat / (np.sqrt(v_W1_hat) + epsilon)
        l1_bias -= learning_rate * m_B1_hat / (np.sqrt(v_B1_hat) + epsilon)
        l2_weights -= learning_rate * m_W2_hat / (np.sqrt(v_W2_hat) + epsilon)
        l2_bias -= learning_rate * m_B2_hat / (np.sqrt(v_B2_hat) + epsilon)

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss[0]}')

# After training, you can use the trained network to make predictions.
# For example, if you have a test input X_test, you can perform forward propagation as shown earlier.

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
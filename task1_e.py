import numpy as np
import matplotlib.pyplot as plt

# Define the training data
x_train = np.arange(-1, 1, 0.05)
d_train = 0.8 * x_train**3 + 0.3 * x_train**2 - 0.4 * x_train + np.random.normal(0, 0.02, len(x_train))

# Initialize network parameters
np.random.seed(42)
input_size = 1
hidden_size = 3
output_size = 1

W1 = np.random.randn(input_size, hidden_size)
B1 = np.random.randn(1, hidden_size)
W2 = np.random.randn(hidden_size, output_size)
B2 = np.random.randn(1, output_size)

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
num_epochs = 1000

for epoch in range(num_epochs):
    for i in range(len(x_train)):
        # Forward propagation
        X = x_train[i:i+1]
        Z1 = np.dot(X, W1) + B1
        A1 = np.tanh(Z1)
        Z2 = np.dot(A1, W2) + B2
        Y = Z2

        # Compute loss
        loss = 0.5 * (Y - d_train[i])**2

        # Backpropagation
        delta2 = Y - d_train[i]
        dW2 = np.dot(A1.T, delta2)
        dB2 = delta2
        delta1 = np.dot(delta2, W2.T) * (1 - A1**2)
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
        W1 -= learning_rate * m_W1_hat / (np.sqrt(v_W1_hat) + epsilon)
        B1 -= learning_rate * m_B1_hat / (np.sqrt(v_B1_hat) + epsilon)
        W2 -= learning_rate * m_W2_hat / (np.sqrt(v_W2_hat) + epsilon)
        B2 -= learning_rate * m_B2_hat / (np.sqrt(v_B2_hat) + epsilon)

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss[0]}')

# After training, you can use the trained network to make predictions.
# For example, if you have a test input X_test, you can perform forward propagation as shown earlier.

x_test = np.arange(-0.97, 0.93, 0.1)
y_test = 0.8* x_test**3 + 0.3 * x_test**2 - 0.4*x_test + np.random.normal(0, 0.02, len(x_test))
predictions = []
for i in range(len(x_test)):
    l1_input = np.dot(x_test[i], W1) + B1
    l1_activated = np.tanh(l1_input)
    output = np.sum(np.dot(l1_activated, W2) + B2)
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
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate some random training data
X_train = np.arange(-1, 1, 0.05)
Y_train = 0.8* X_train**3 + 0.3 * X_train**2 - 0.4*X_train + np.random.normal(0, 0.02, len(X_train))

# Build the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation="tanh", input_shape=(1,)),
    tf.keras.layers.Dense(1, activation="linear")
])

# Compile the model with SGD optimizer and mean squared error loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

model.compile(optimizer=optimizer, loss='mean_squared_error')

# Print a summary of the model architecture
model.summary()

# Train the model on the training data
history = model.fit(X_train, Y_train, epochs=1000, verbose=1)

# Test the model with some sample data
X_test = np.arange(-1, 1, 0.05)
predictions = model.predict(X_test)

# Plot the training data, true cubic function, and predictions
plt.figure(figsize=(8, 6))
plt.scatter(X_train, Y_train, color='blue', label='Training Data')
plt.plot(X_test, predictions, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=8, label='Predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Cubic Function Approximation using Neural Network')
plt.legend()
plt.grid(True)
plt.show()

# Plot the loss curve during training
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], color='blue', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
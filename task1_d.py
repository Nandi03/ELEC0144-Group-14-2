from network import Model, Layer
import numpy as np
import matplotlib.pyplot as plt

# set seed value for reproducability
np.random.seed(42)

# Instantiate a Model object with its hyperparameters
model = Model(learning_rate=0.1, optimizer="sgd")

# Building the neural network and adding on its layers with the activation functions

# Task 1d - modify the activation functions for each layer by modifying str value of 'activation_function'
# e.g. to change the first hidden layer to use relu:
#       model.layers.append(Layer("relu", 1, 3)) # use string 'relu'
# other activation functions including, sigmoid, leaky_relu, linear and tanh.

model.layers.append(Layer("relu", 1, 3))
model.layers.append(Layer("linear", 3, 1))

# Creating the training data set
x_train = np.arange(-1, 1, 0.05)
d_train = 0.8 * x_train**3 + 0.3 * x_train**2 - 0.4 * x_train + np.random.normal(0, 0.02, len(x_train)) # add randomness

# train the network
model.compile(x_train, d_train)

# Create the testing data set
x_test = np.arange(-0.97, 0.93, 0.1)
y_test = 0.8* x_test**3 + 0.3 * x_test**2 - 0.4*x_test 

# Make predictions using the model after training
predictions = model.fit(x_test, y_test)

# PLOTTING GRAPHS
# Regression results as a line graph
# showing the x_test values on the x-axis and predicted values on the y-axis
plt.figure(figsize=(8, 6))
plt.scatter(x_test, y_test, color='blue', label='Training Data')
plt.plot(x_test, predictions, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=8, label='Predictions')
plt.xlabel('unseen data inputs')
plt.ylabel('outputs')
plt.title('Cubic regression for 1-3-1 NN')
plt.legend()
plt.grid(True)
plt.show()

# Plot the training loss curve
# Showing the number of epochs on the x-axis and the cost at that epoch on the y-axis
# - useful for analysing convergence during training and other features of the training algorithm
plt.figure(figsize=(8, 6))
plt.plot(model.history['train'], color='blue', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Cost (using squared error)')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()


# Plot the testing loss curve
# Showing the x_test on the x-axis and the loss calculated on the y-axis
# - useful for analysing generalisation 
plt.figure(figsize=(8, 6))
plt.plot(model.history['test'], color='blue', label='Testing Loss')
plt.xlabel('Sample Number')
plt.ylabel('Squared Error')
plt.title('Testing Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

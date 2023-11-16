from network import Model, Layer
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
model = Model(learning_rate=0.0042, optimizer="adam")
#model.momentum = 0.3

model.layers.append(Layer("tanh", 1, 3))
model.layers.append(Layer("linear", 3, 1))

x_train = np.arange(-1, 1, 0.05)
num_elements = int(0.8 * len(x_train)) 
selected_indices = np.random.choice(len(x_train), num_elements, replace=False)
# x_train = x_train[selected_indices] # uncomment to test with different sized training data
d_train = 0.8 * x_train**3 + 0.3 * x_train**2 - 0.4 * x_train + np.random.normal(0, 0.02, len(x_train))

model.compile(x_train, d_train)

x_test = np.arange(-0.97, 0.93, 0.1)
y_test = 0.8* x_test**3 + 0.3 * x_test**2 - 0.4*x_test 
predictions = model.fit(x_test)

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

plt.figure(figsize=(8, 6))
plt.plot(model.history, color='blue', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
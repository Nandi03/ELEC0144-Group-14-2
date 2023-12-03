from network import Model, Layer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

np.random.seed(42)

# Read data from the text file into a pandas DataFrame
data = pd.read_csv('IrisData.txt', header=None, names=['col1', 'col2', 'col3', 'col4', 'class'])

# Create a mapping between unique string labels and integers
label_to_int = {label: idx for idx, label in enumerate(data['class'].unique())}

# Map string labels to integers and create a new column 'class_int' in the DataFrame
data['class_int'] = data['class'].map(label_to_int)

# Extract numerical features and integer labels
data_x = np.array(data[['col1', 'col2', 'col3', 'col4']].values)
data_y = np.array(data['class_int'].values)

combined_data = list(zip(data_x, data_y))
np.random.shuffle(combined_data)
data_x, data_y = zip(*combined_data)
data_x= np.array(data_x)
data_y = np.array(data_y)

total_samples = len(data_x)
train_size = int(0.7 * total_samples)

# Splitting the data into training and testing sets
x_train, x_test = data_x[:train_size], data_x[train_size:]
y_train, y_test = data_y[:train_size], data_y[train_size:]

# Using Stochastic Gradient Descent (SGD) with Momentum

# using 1000 epochs
model = Model(learning_rate=0.1, optimizer="sgd_momentum", epochs=1000)
#model = Model(learning_rate=0.01, optimizer="sgd_momentum", epochs=1000)
#model = Model(learning_rate=0.001, optimizer="sgd_momentum", epochs=1000)

# using 5000 epochs
#model = Model(learning_rate=0.01, optimizer="sgd_momentum", epochs=5000) # optimal for SGD + Momentum
#model.momentum = 0.4

# Using SGD with Adaptive Gradient (AdaGrad)

# Using 1000 epochs
#model = Model(learning_rate=0.1, optimizer="sgd_adaptive", epochs=1000)
#model = Model(learning_rate=0.01, optimizer="sgd_adaptive", epochs=1000)

# Using 5000 epochs
#model = Model(learning_rate=0.01, optimizer="sgd_adaptive", epochs=5000) # optimal for SGD + AdaGrad

# Using Adam

# Using 1000 epochs
#model = Model(learning_rate=0.1, optimizer="Adam", epochs=1000)
#model = Model(learning_rate=0.01, optimizer="Adam", epochs=1000)
#model = Model(learning_rate=0.001, optimizer="Adam", epochs=1000)

# Using 10000 epochs
#model = Model(learning_rate=0.001, optimizer="Adam", epochs=10000)

model.layers.append(Layer("tanh", 4, 5))
model.layers.append(Layer("tanh", 5, 3))
model.layers.append(Layer("linear", 3, 3))

model.compile(x_train, y_train)

predictions = model.fit(x_test, y_test)
predictions = [round(max(arr[0])) for arr in predictions]
# Plot the training data, true cubic function, and predictions
plt.figure(figsize=(8, 6))
plt.plot(y_test, color='red', label='Actual')
plt.plot(predictions, color='blue', label="Predicted")
plt.xlabel('Sample Number')
plt.ylabel('Class')
plt.title('Classification results')
plt.legend()
plt.grid(True)
plt.show()

# Plot confusion matrix
cm = confusion_matrix(y_test, predictions)
num_classes = 3
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

error = y_test - predictions
plt.figure(figsize=(8, 6))
plt.plot(error,  color='red', label='Error')
plt.xlabel('Sample Number')
plt.ylabel('Error')
plt.title('Error between predictions and actual values')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(model.history['train'], color='blue', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Cost (using squared error)')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(model.history['test'], color='blue', label='Testing Loss')
plt.xlabel('Sample Number')
plt.ylabel('Squared Error')
plt.title('Testing Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

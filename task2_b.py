from network import Model, Layer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# set seed value for reproducability
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

# Shuffle the data set for better generalisation and avoid overfitted models
combined_data = list(zip(data_x, data_y))
np.random.shuffle(combined_data)
data_x, data_y = zip(*combined_data)
data_x= np.array(data_x)
data_y = np.array(data_y)

# take the first 70% of the shuffled data for training
total_samples = len(data_x)
train_size = int(0.7 * total_samples)

# Splitting the data into training and testing sets
x_train, x_test = data_x[:train_size], data_x[train_size:]
y_train, y_test = data_y[:train_size], data_y[train_size:]

# uncomment the instantiation of the Model with the hyperparameters to test

# using 1000 epochs
#model = Model(learning_rate=0.1, optimizer="sgd", one_hot=True, epochs=1000)
#model = Model(learning_rate=0.01, optimizer="sgd", one_hot=True, epochs=1000)
model = Model(learning_rate=0.001, optimizer="sgd", one_hot=True, epochs=5000)
#model = Model(learning_rate=0.0001, optimizer="sgd", one_hot=True, epochs=1000)

# using 20000 epochs
#model = Model(learning_rate=0.001, optimizer="sgd", one_hot=True, epochs=20000) 

# using 15000 epochs
model = Model(learning_rate=0.001, optimizer="sgd", one_hot=True, epochs=15000)

# using 10000 epochs
#model = Model(learning_rate=0.001, optimizer="sgd", one_hot=True, epochs=10000) # optimal

# using 5000 epochs
#model = Model(learning_rate=0.001, optimizer="sgd", one_hot=True, epochs=5000)

# Without one-hot encoding
#model = Model(learning_rate=0.1, optimizer="sgd", one_hot=False, epochs=1000)

# Building the neural network and adding on its layers with the activation functions
model.layers.append(Layer("tanh", 4,3))
model.layers.append(Layer("tanh", 3, 3))
model.layers.append(Layer("linear", 3, 3))

# train the model
model.compile(x_train, y_train)

# Make predictions using the model after training
predictions = model.fit(x_test, y_test)
predictions = np.argmax(np.array([arr[0] for arr in predictions]), axis=1) # use this when using one-hot encoding
#predictions = [round(i) for i in predictions] # use this when not using one-hot encoding.

# PLOTTING GRAPHS
# Clasification results as a line graph
# showing the sample number of the x-axis and class on the y-axis
# - for sample specific analysis
plt.figure(figsize=(8, 6))
plt.plot(y_test, color='red', label='Actual')
plt.plot(predictions, color='blue', label="Predicted")
plt.xlabel('Sample Number')
plt.ylabel('Class')
plt.title('Classification results')
plt.legend()
plt.grid(True)
plt.show()

# Plot a confusion matrix
# Showing the number of correct and incorrect predictions class 
# - for class specific analysis
cm = confusion_matrix(y_test, predictions)
num_classes = 3
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot an error graph of the error between the actual and predicted values
# Shows the sample number of the testing data set on the x-axis and the error on the y-axis
# - for sample specific analysis
error = y_test - predictions
plt.figure(figsize=(8, 6))
plt.plot(error,  color='red', label='Error')
plt.xlabel('Sample Number')
plt.ylabel('Error')
plt.title('Error between predictions and actual values')
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
# Showing the sample number on the x-axis and the loss calculated on the y-axis
# - useful for analysing generalisation or for sample specific analysis
plt.figure(figsize=(8, 6))
plt.plot(model.history['test'], color='blue', label='Testing Loss')
plt.xlabel('Sample Number')
plt.ylabel('Squared Error')
plt.title('Testing Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

from network import Model, Layer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(40)

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

model = Model(learning_rate=0.0000033, optimizer="sgd_adaptive", classification=True)
#model.momentum = 0.3

model.layers.append(Layer("tanh", 4, 5))
model.layers.append(Layer("tanh", 5, 3))
model.layers.append(Layer("linear", 3, 3))


model.compile(x_train, y_train)

predictions = model.fit(x_test, y_test)

predicted_classes = np.round(np.argmax(np.array([arr[0] for arr in predictions]), axis=1))
# Plot the training data, true cubic function, and predictions
plt.figure(figsize=(8, 6))
plt.plot(y_test, color='red', label='Actual')
plt.plot(predicted_classes, color='blue', label="Predicted")
plt.xlabel('Sample Number')
plt.ylabel('Class')
plt.title('Classification results')
plt.legend()
plt.grid(True)
plt.show()

error = y_test - predicted_classes
plt.figure(figsize=(8, 6))
plt.plot(error,  color='red', label='Error')
plt.xlabel('Sample Number')
plt.ylabel('Error')
plt.title('Error')
plt.legend()
plt.grid(True)
plt.show()
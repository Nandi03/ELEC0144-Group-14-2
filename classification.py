import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2


np.random.seed(42)

# Initialize weights randomly
np.random.seed(42)
input_size = 4
hidden_size = 5
hidden2_size = 3
output_size = 3

l1_weights = np.random.randn(input_size, hidden_size) * np.sqrt(2 / (input_size + hidden_size))
l1_bias = np.zeros((1, hidden_size))  # Bias for hidden layer
l2_weights = np.random.randn(hidden_size, hidden2_size) * np.sqrt(2 / (hidden_size + hidden2_size))
l2_bias = np.zeros((1, hidden2_size))  
l3_weights = np.random.randn(hidden2_size, output_size) * np.sqrt(2 / (hidden2_size + output_size))
l3_bias = np.zeros((1, output_size))  # Bias for output layer


# Learning rate
learning_rate = 0.01

# Number of epochs
epochs = 1000

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


for epoch in range(epochs):
    for i in range(len(x_train)):
        x = x_train[i]
        Z1 = np.dot(x, l1_weights) + l1_bias
        A1 = np.tanh(Z1)
        Z2 = np.dot(A1, l2_weights) + l2_bias
        A2 = np.tanh(Z2)
        output = np.dot(A2, l3_weights) + l3_bias

        loss = 

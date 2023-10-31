
import numpy as np
import pandas as pd

# Step 1: Import data
data = pd.read_csv('IrisData.txt', header=None)
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = data.sample(frac=1).reset_index(drop=True)  # Shuffle the data

# Step 2: Preprocess data
# Use only sepal_length and sepal_width
X = data[['sepal_length', 'sepal_width']].values
# Convert class labels to one-hot encoding
y = pd.get_dummies(data['class']).values

# Step 3: Modify network for 2-3-3 architecture

# Initialize weights and biases
input_dim = 2
hidden_dim = 3
output_dim = 3

weights = {
    'wh': np.random.randn(input_dim, hidden_dim),
    'bh': np.random.randn(hidden_dim),
    'wy': np.random.randn(hidden_dim, output_dim),
    'by': np.random.randn(output_dim)
}
def tanh_prime(x):
    return 1 - np.tanh(x)**2

def forward(x, weights):
    h = np.tanh(np.dot(x, weights['wh']) + weights['bh'])
    y = np.dot(h, weights['wy']) + weights['by']
    return h, softmax(y)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def backpropagation(x, h, y_pred, y_true, weights, lr):
    # Gradient of Loss with respect to output y_pred
    dL_dy = y_pred - y_true
    
    # Gradients for output layer
    dL_dwy = np.outer(h, dL_dy)
    dL_dby = dL_dy
    
    # Gradients for hidden layer
    dL_dh = np.dot(weights['wy'], dL_dy)
    dL_dwh = np.outer(x, dL_dh * tanh_prime(np.dot(x, weights['wh']) + weights['bh']))
    dL_dbh = dL_dh * tanh_prime(np.dot(x, weights['wh']) + weights['bh'])
    
    # Update weights
    weights['wh'] -= lr * dL_dwh
    weights['bh'] -= lr * dL_dbh
    weights['wy'] -= lr * dL_dwy
    weights['by'] -= lr * dL_dby

    return weights

# Training loop
lr = 0.01
epochs = 1000
for epoch in range(epochs):
    total_loss = 0
    for xi, yi in zip(X, y):
        h, y_pred = forward(xi, weights)
        total_loss += -np.sum(yi * np.log(y_pred))
        weights = backpropagation(xi, h, y_pred, yi, weights, lr)
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {total_loss/len(X)}')

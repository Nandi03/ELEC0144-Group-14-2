import numpy as np
import math
from tsensor import explain as exp

np.random.seed(0)
class Model:
    def __init__(self, epochs=1000, learning_rate=0.1, optimizer="sgd", classification=False) -> None:
        '''
        epochs: int
        learning_rate: float 0-1
        optimizer: sgd, adam, sgd-momentum, sgd-adaptive
        '''
        self.layers = []
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.history = []  # training loss
        self.betas = [0.9, 0.99]
        self.epsilon = 1e-8
        self.momentum = 0.5 # a constant between 0 and 1
        self.classification = classification
        

    def cross_entropy(self, actual, predicted):
        actual_one_hot = np.eye(len(predicted[0]))[int(actual)]
        predicted = np.clip(predicted, 0.1, 0.9)  # Clip to avoid numerical instability
        loss = -np.sum(actual_one_hot * np.log(predicted))
        loss /= len(predicted[0])
        return np.array([loss])

    def cross_entropy_grad(self, actual, predicted):
        actual_one_hot = np.eye(len(predicted[0]))[int(actual)]
        predicted = np.clip(predicted, 0.1, 0.9)  # Clip to avoid numerical instability
        gradients = -actual_one_hot / predicted
        return gradients
    
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
        

    def compile(self, x, y):
        if self.optimizer == "sgd":
           self.sgd(x, y)
        if self.optimizer == "adam":
           self.adam(x, y)
        if self.optimizer == "sgd_momentum":
            self.sgd_momentum(x, y)
        if self.optimizer == "sgd_adaptive":
            self.sgd_adaptive(x, y)
        

    def sgd(self, x, y):
        for epoch in range(self.epochs):
            for i in range(len(x)):
                # Forward pass
                output, output_deactivated, input = self.forward(x[i])
            
                # backpropagation
                loss = None
                if self.classification:
                    loss = self.cross_entropy(y[i], output)
                else:
                    loss = self.mse(y[i], output)
                                    
                output_grad = None
    
                for j in range(len(self.layers) - 1, -1, -1):
                    if j == len(self.layers) - 1:
                        if not self.classification:
                            output_grad = self.mse_grad(y[i], output) * self.layers[j].get_derivative(output_deactivated[j])
                        else:
                            output_grad = self.cross_entropy_grad(y[i], output) * self.layers[j].get_derivative(output_deactivated[j])                   

                    else:
                        output_grad = np.dot(output_grad, self.layers[j+1].weights.T) * self.layers[j].get_derivative(output_deactivated[j])

                    self.layers[j].weights -= self.learning_rate * np.outer(input[j], output_grad) 
                    self.layers[j].bias -= self.learning_rate * output_grad

            self.history.append(float(loss[0]))

    def adam(self, x, y):
        if len(self.betas) != 2:
            raise IndexError
        for epoch in range(self.epochs):
            for i in range(len(x)):
                output, output_deactivated, input = self.forward(x[i])

                # Back propagation
                loss = None
                if self.classification:
                    loss = self.cross_entropy(y[i], output)
                else:
                    loss = self.mse(y[i], output)

                output_grad = None
                for j in range(len(self.layers)-1, -1, -1):
                    if j == len(self.layers) - 1:
                        if not self.classification:
                            output_grad = self.mse_grad(y[i], output) * self.layers[j].get_derivative(output_deactivated[j])
                        else:
                            output_grad = self.cross_entropy_grad(y[i], output) * self.layers[j].get_derivative(output_deactivated[j])                   
                    else:
                        output_grad = np.dot(output_grad, self.layers[j+1].weights.T) * self.layers[j].get_derivative(output_deactivated[j])
                    dW = np.dot(input[j].T, output_grad)
                    dB = output_grad

                    # update adam variables
                    self.layers[j].m_W = self.betas[0] * self.layers[j].m_W + (1 - self.betas[0]) * dW
                    self.layers[j].m_B = self.betas[0] * self.layers[j].m_B + (1 - self.betas[0]) * dB
                    self.layers[j].v_W = self.betas[1] * self.layers[j].v_W + (1 - self.betas[1]) * (dW**2)
                    self.layers[j].v_B = self.betas[1] * self.layers[j].v_B + (1 - self.betas[1]) * (dB**2)
                    # bias correction
                    m_W1_hat = self.layers[j].m_W  / (1 - self.betas[0]**(i+1))
                    m_B1_hat = self.layers[j].m_B / (1 - self.betas[0]**(i+1))
               
                    v_W1_hat = self.layers[j].v_W  / (1 - self.betas[1]**(i+1))
                    v_B1_hat =self.layers[j].v_B / (1 - self.betas[1]**(i+1))
                  
                    self.layers[j].weights -= self.learning_rate * m_W1_hat / (np.sqrt(v_W1_hat) + self.epsilon)
                    self.layers[j].bias -= self.learning_rate * m_B1_hat / (np.sqrt(v_B1_hat) + self.epsilon)
            self.history.append(float(loss[0]))

    def sgd_momentum(self, x, y):
        for epoch in range(self.epochs):
            for i in range(len(x)):
                # Forward pass
                output, output_deactivated, input = self.forward(x[i])
            
                # Backpropagation
                loss = None
                if self.classification:
                    loss = self.cross_entropy(y[i], output)
                else:
                    loss = self.mse(y[i], output)
                output_grad = None

                for j in range(len(self.layers) - 1, -1, -1):
                    if j == len(self.layers) - 1:
                        if not self.classification:
                            output_grad = self.mse_grad(y[i], output) * self.layers[j].get_derivative(output_deactivated[j])
                        else:
                            output_grad = self.cross_entropy_grad(y[i], output) * self.layers[j].get_derivative(output_deactivated[j])                   
                    else:
                        output_grad = np.dot(output_grad, self.layers[j+1].weights.T) * self.layers[j].get_derivative(output_deactivated[j])

                    # Update velocities with momentum
                    self.layers[j].velocity = self.momentum * self.layers[j].velocity + self.learning_rate * np.outer(input[j], output_grad)
                    self.layers[j].bias_velocity = self.momentum * self.layers[j].bias_velocity + self.learning_rate * output_grad

                    # Update weights and biases using momentum
                    self.layers[j].weights -= self.layers[j].velocity
                    self.layers[j].bias -= self.layers[j].bias_velocity

            self.history.append(float(loss[0]))

        
    def sgd_adaptive(self, x, y):
        for epoch in range(self.epochs):
            for i in range(len(x)):
                # Forward pass
                output, output_deactivated, input = self.forward(x[i])

                # Backpropagation
                loss = None
                if self.classification:
                    loss = self.cross_entropy(y[i], output)
                else:
                    loss = self.mse(y[i], output)
                output_grad = None

                for j in range(len(self.layers) - 1, -1, -1):
                    if j == len(self.layers) - 1:
                        if not self.classification:
                            output_grad = self.mse_grad(y[i], output) * self.layers[j].get_derivative(output_deactivated[j])
                        else:
                            output_grad = self.cross_entropy_grad(y[i], output) * self.layers[j].get_derivative(output_deactivated[j])                   
                    else:
                        output_grad = np.dot(output_grad, self.layers[j+1].weights.T) * self.layers[j].get_derivative(output_deactivated[j])

                    # Accumulate squared gradients for Adagrad
                    self.layers[j].velocity += output_grad**2
                    self.layers[j].bias_velocity  += np.sum(output_grad**2)

                    # Update weights and biases using Adagrad
                    self.layers[j].weights -= (self.learning_rate / (np.sqrt(self.layers[j].velocity) + self.epsilon)) * np.outer(input[j], output_grad)
                    self.layers[j].bias -= (self.learning_rate / (np.sqrt(self.layers[j].bias_velocity) + self.epsilon)) * output_grad

            self.history.append(float(loss[0]))

    def forward(self, x):
        # Forward pass
        output = None
        output_deactivated = []
        input = [x]
        for j in range(len(self.layers)):
            output = (np.dot(input[j], self.layers[j].weights) + self.layers[j].bias) 
            output_deactivated.append(output)
            output = self.layers[j].get_activation(output)
            input.append(output)
        return output, output_deactivated, input
    
    def mse(self, actual, predicted):
        return 0.5 * ((predicted- actual)**2)

    def mse_grad(self, actual, predicted):
        return predicted - actual

    def fit(self, x, y):
        predictions = []
        for i in range(len(x)):
            input = x[i]
            output = None
            for j in range(len(self.layers)):
                output = (np.dot(input, self.layers[j].weights) + self.layers[j].bias) 
                output_activated = self.layers[j].get_activation(output)
                input = output_activated
                if len(output_activated[0]) == 1:
                    output_activated = np.sum(output_activated)
                
            predictions.append(output_activated)


        return predictions


class Layer:
    def __init__(self, activation="linear", input_shape=1, output_shape=1):
        self.activation = activation
        self.weights = np.random.randn(input_shape, output_shape) * np.sqrt(2 / (input_shape + output_shape))
    
        self.bias = np.zeros((1, output_shape)) 
        # for adam 
        self.m_W, self.m_B = 0, 0
        self.v_W, self.v_B = 0, 0
        # for sgd + momentum
        self.velocity = np.zeros_like(self.weights)
        self.bias_velocity = np.zeros_like(self.bias)


    def get_activation(self, x, alpha=0.1):
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        
        elif self.activation == "relu":
            return np.maximum(0, x)
        
        elif self.activation == "linear":
            return x
        
        elif self.activation == "leaky_relu":
            return np.where(x > 0, x, alpha * x)
        
        elif self.activation == "tanh":
            return np.tanh(x)
        
        elif self.activation == "softmax":
            e_x = np.exp(x - np.max(x))  # Avoid numerical instability by subtracting the maximum value
            return e_x / e_x.sum(axis=-1, keepdims=True)

        raise ValueError("Invalid Activation ")



    def get_derivative(self, x, alpha=0.1):

        if self.activation == "sigmoid":
            return self.get_activation(x) * (1 - self.get_activation(x))
        
        elif self.activation == "relu":
            return np.where(x > 0, 1, 0)
        
        elif self.activation == "linear":
            return 1
        
        elif self.activation == "leaky_relu":
            return np.where(x > 0, 1, alpha)
        
        elif self.activation == "tanh":
            return 1.0 - np.tanh(x)**2
        
        elif self.activation == "softmax":
            softmax_probs = self.get_activation(x)
            d_softmax = softmax_probs * (1 - softmax_probs)
            return d_softmax
        

        

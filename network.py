import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
class Model:
    def __init__(self, epochs=1000, learning_rate=0.1, optimizer="sgd") -> None:
        '''
        epochs: int
        learning_rate: float 0-1
        optimizer: sgd, adam
        '''
        self.layers = []
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.history = []  # training loss
        self.betas = [0.9, 0.99]
        self.epsilon = 1e-8


    def compile(self, x, y):
        if self.optimizer == "sgd":
           self.sgd(x, y)
        if self.optimizer == "adam":
           self.adam(x, y)

        if self.optimizer == "newton":
            self.newton_gauss(x, y)

    def sgd(self, x, y):
        for epoch in range(self.epochs):
            for i in range(len(x)):
                # Forward pass
                output, output_deactivated, input = self.forward(x[i])
            
                # backpropagation
                loss = self.mse(y[i], output)
                output_grad = None
    
                for j in range(len(self.layers) - 1, -1, -1):
                    if j == len(self.layers) - 1:
                        output_grad = self.mse_grad(y[i], output) * self.layers[j].get_derivative(output_deactivated[j])
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
                loss = self.mse(y[i], output)

                output_grad = None
                for j in range(len(self.layers)-1, -1, -1):
                    if j == len(self.layers) - 1:
                        output_grad = self.mse_grad(y[i], output) * self.layers[j].get_derivative(output_deactivated[j])
                        
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
            predictions.append(np.sum(output_activated))


        return predictions


class Layer:
    def __init__(self, nodes=1, activation="linear", input_shape=1, output_shape=1, beta=0.9):
        self.nodes = nodes
        self.activation = activation
        self.shape = [input_shape, nodes]
        self.weights = np.random.randn(input_shape, output_shape) * np.sqrt(2 / (input_shape + output_shape))
        self.bias = np.zeros((1, output_shape))  
        self.m_W, self.m_B = 0, 0
        self.v_W, self.v_B = 0, 0


    def get_activation(self, x, alpha=0.1):
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        
        if self.activation == "relu":
            return np.maximum(0, x)
        
        if self.activation == "linear":
            return x
        
        if self.activation == "leaky_relu":
            return np.where(x > 0, x, alpha * x)
        
        if self.activation == "tanh":
            return np.tanh(x)

        raise ValueError



    def get_derivative(self, x, alpha=0.1):

        if self.activation == "sigmoid":
            return x * (1 - x)
        
        if self.activation == "relu":
            return np.where(x > 0, 1, 0)
        
        if self.activation == "linear":
            return 1
        
        if self.activation == "leaky_relu":
            return np.where(x > 0, 1, alpha)
        
        if self.activation == "tanh":
            return 1.0 - np.tanh(x)**2
        

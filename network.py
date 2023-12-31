import numpy as np
np.random.seed(42) # seed value for reproducability

class Model:
    '''
    Implements functions to create and train a neural network.
    :param epochs: int, number of epochs (optional)
    :param learning_rate: float, a value between 0 - 1 (optional)
    :param optimizer: str, a choice between sgd, adam, sgd-momentum, sgd-adaptive (optional)
    
    Features 4 different training methods:

    > Stochastic Gradient Descent = "sgd" - selected by default.

    > Stochastic Gradient Descent with Momentum = "sgd-momentum"
        - to adjust momentum modify the model.momentum instance variable

    > Stochastic Gradient Descent with AdaGrad = "sgd-adaptive"
        - to adjust beta values modify the model.betas[0] and model.betas[1] instance variables
        - same beta values are used in Adam

    > Adam = "adam"
    
    Other Attributes:
    > layers = []
        - An array of Layer instances. Append new Layers in the order required for the neural network. 
    > betas = [0.9, 0.99]
         - used for training methods Adam, sgd-adaptive
    > epsilon = 1e-8
         - used for sgd_adaptive, adam to avoid square rooting with a value 0.
    > momentum
        - used for sgd-momentum
    > history
        - a dictionary which stores the progress of training by the loss for training and testing
        - access the loss for every epoch during training with self.history['train']
        - access the loss for every sample in testing set with self.history['test']
    
    *iterative equations used in the training methods are found in the report.
    
    '''
    def __init__(self, epochs=1000, learning_rate=0.1, optimizer="sgd", one_hot=False) -> None:
        '''
        Create a neural network.
        Intialised with no layers. Append new layers in the order from input layer to output layer in the list 'layers'.
        '''
        self.layers = []
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.history = {'train':[], 'test':[]}  # training loss
        self.betas = [0.9, 0.99] # 1st value is beta_m, 2nd value is beta_v
        self.epsilon = 1e-8
        self.momentum = 0.5 # a constant between 0 and 1
        self.one_hot = one_hot

    def compile(self, x, d):
        ''' Call the training algorithm for the corresponding optimizer'''
        if self.optimizer == "sgd":
           self.sgd(x, d)
        if self.optimizer == "adam":
           self.adam(x, d)
        if self.optimizer == "sgd_momentum":
            self.sgd_momentum(x, d)
        if self.optimizer == "sgd_adaptive":
            self.sgd_adaptive(x, d)

    def sgd(self, x, d):
        '''
        Train the model using Stochastic Gradient Descent.
        '''
        for epoch in range(self.epochs):
            cost = 0
            # for each sample in the training set - sequential learning
            for i in range(len(x)):
                # Forward pass
                y, v_j, input = self.forward(x[i])
            
                # backpropagation
                loss = self.squared_error(d[i], y) # calculate the loss
                                    
                output_grad = None

                # loop through each of the layers in the network and update its weights/biases
                for j in range(len(self.layers) - 1, -1, -1):
                    # find the rate of change of the weights using the iterative equations
                    # partial derivative of the loss wrt the input of the neurons
                    if j == len(self.layers) - 1:
                        output_grad = self.squared_error_grad(d[i], y) * self.layers[j].get_derivative(v_j[j])                   
                    else:
                        output_grad = np.dot(output_grad, self.layers[j+1].weights.T) * self.layers[j].get_derivative(v_j[j])
                    
                    # update the weights and bias
                    self.layers[j].weights -= self.learning_rate * np.outer(input[j], output_grad) 
                    self.layers[j].bias -= self.learning_rate * output_grad

                cost += float(np.sum(loss))

            cost /= len(x) # average loss
            # Append cost every epoch for plotting and tracking learning progress
            self.history['train'].append(cost)

    def adam(self, x, d):
        '''
        Train the model using Adam.
        '''
        if len(self.betas) != 2:
            raise IndexError
        for epoch in range(self.epochs):
            cost = 0
            # for each sample in the training set - sequential learning
            for i in range(len(x)):
                # forward propagation
                y, v_j, input = self.forward(x[i])

                # Back propagation
                loss = self.squared_error(d[i], y)

                output_grad = None

                # loop through each layer and update its weights/ biases
                for j in range(len(self.layers)-1, -1, -1):
                    # find the rate of change of the weights using the iterative equations
                    # partial derivative of the loss wrt the input of the neurons
                    if j == len(self.layers) - 1:
                        output_grad = self.squared_error_grad(d[i], y) * self.layers[j].get_derivative(v_j[j])                   
                    else:
                        output_grad = np.dot(output_grad, self.layers[j+1].weights.T) * self.layers[j].get_derivative(v_j[j])
                  
                    dW = np.dot(input[j].T.reshape(-1, 1), output_grad)
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

                    # update weights and biases
                    self.layers[j].weights -= self.learning_rate * m_W1_hat / (np.sqrt(v_W1_hat + self.epsilon))
                    self.layers[j].bias -= self.learning_rate * m_B1_hat / (np.sqrt(v_B1_hat + self.epsilon))
                
                cost += float(np.sum(loss))
            cost /= len(x) # average loss 
            # Append cost every epoch for plotting and tracking learning progress
            self.history['train'].append(cost)

    def sgd_momentum(self, x, d):
        '''
        Train the model using Stochastic Gradient Descent with Momentum.
        '''
        for epoch in range(self.epochs):
            cost = 0
            for i in range(len(x)):
                # Forward pass
                y, v_j, input = self.forward(x[i])
            
                # Backpropagation
                loss = self.squared_error(d[i], y)
                output_grad = None

                # looping through each layer and updating its weight and biases
                for j in range(len(self.layers) - 1, -1, -1):
                    # find the rate of change of the weights using the equations
                    # partial derivative of the loss wrt the input of the neurons
                    if j == len(self.layers) - 1:
                        output_grad = self.squared_error_grad(d[i], y) * self.layers[j].get_derivative(v_j[j])                   
                    else:
                        output_grad = np.dot(output_grad, self.layers[j+1].weights.T) * self.layers[j].get_derivative(v_j[j])

                    # Update velocities with momentum
                    self.layers[j].velocity = self.momentum * self.layers[j].velocity +  np.outer(input[j], output_grad)
                    self.layers[j].bias_velocity = self.momentum * self.layers[j].bias_velocity + output_grad

                    # Update weights and biases using momentum
                    self.layers[j].weights -= self.learning_rate * self.layers[j].velocity
                    self.layers[j].bias -= self.learning_rate * self.layers[j].bias_velocity
                cost += float(np.sum(loss))
            cost /= len(x) # average loss
            # Append cost every epoch for plotting and tracking learning progress
            self.history['train'].append(cost)

        
    def sgd_adaptive(self, x, d):
        '''
        Train the model using Stochastic Gradient Descent with Adaptive Learning Rate.
        '''
        for epoch in range(self.epochs):
            cost = 0
            for i in range(len(x)):
                # Forward pass
                y, v_j, input = self.forward(x[i])

                # Backpropagation
                loss = self.squared_error(d[i], y)
                output_grad = None

                for j in range(len(self.layers) - 1, -1, -1):
                    # find the rate of change of the weights using the equations
                    # partial derivative of the loss wrt the input of the neurons
                    if j == len(self.layers) - 1:
                        output_grad = self.squared_error_grad(d[i], y) * self.layers[j].get_derivative(v_j[j])                   
                    else:
                        output_grad = np.dot(output_grad, self.layers[j+1].weights.T) * self.layers[j].get_derivative(v_j[j])
                    
                    # Accumulate squared gradients for Adagrad
                    self.layers[j].velocity += output_grad**2
                    self.layers[j].bias_velocity  += np.sum(output_grad**2)

                    # Update weights and biases using Adagrad
                    self.layers[j].weights -= (self.learning_rate / (np.sqrt(self.layers[j].velocity+ self.epsilon))) * np.outer(input[j], output_grad)
                    self.layers[j].bias -= (self.learning_rate / (np.sqrt(self.layers[j].bias_velocity + self.epsilon))) * output_grad
                cost += float(np.sum(loss))
            cost /= len(x) # average loss
            # Append cost every epoch for plotting and tracking learning progress
            self.history['train'].append(cost)

    def forward(self, x):
        ''' 
        Run the forward propagation

        Parameters:
        x: a value(s) of type float
        
        Returns:
        > x_out: the output at each layer as a 2D array
        > v_j: the output without activation function at each layer as a 2D array
        > input: 2D array, the input to each layer 
        
        '''
        # Forward propagagtion
        x_out = None
        v_j = [] # stores the output of all neurons without activation, with every layer
        input = [x] # stores the input of all neurons, with every layer
        for s in range(len(self.layers)):
            x_out = (np.dot(input[s], self.layers[s].weights) + self.layers[s].bias) 
            v_j.append(x_out)
            x_out = self.layers[s].get_activation(x_out) # apply activation function
            input.append(x_out) # output of neuron is the input for the next neuron
        return x_out, v_j, input
    
    def squared_error(self, actual, predicted):
        ''' 
        Calculate and return the squared error

        Parameters:
        > actual: the actual (true) x_out(s) as float, doesn't expect one-hot encoding
        > predicted: the predicted value(s) as float(s)
        
        return:
        > derivative of squared error as float(s)
        
        '''
        one_hot = None
        if self.one_hot:
            # translate actual value to one-hot encoding
            one_hot = np.zeros_like(predicted[0])
            one_hot[actual] = 1
            actual = [one_hot]
        return 0.5 * ((predicted- actual)**2)

    def squared_error_grad(self, actual, predicted):
        ''' 
        Calculate and return the derivative of the squared error
        
        Parameters:
        > actual: the actual (true) x_out(s) as float, doesn't expect one-hot encoding
        > predicted: the predicted value(s) as float(s)
        
        returns:
        > derivative of squared error as float(s)

        ''' 
        one_hot = None
        if self.one_hot:
            # translate actual value to one-hot encoding
            one_hot = np.zeros_like(predicted[0])
            one_hot[actual] = 1
            actual = [one_hot]
        
        return predicted - actual

    def fit(self, x, y):
        ''' 
        Given x-values from a testing data set, predicts the d-values using the model after training.

        Parameters:
        > x: an array of the testing values compatible with the model
        
        returns:
        > An array of the predictions (output y)

        '''
        # running forward propagation on each of the samples of the testing set
        predictions = []
        for i in range(len(x)):
            input = x[i]
            x_out = None
            for j in range(len(self.layers)):
                x_out = (np.dot(input, self.layers[j].weights) + self.layers[j].bias) 
                output_activated = self.layers[j].get_activation(x_out)
                input = output_activated
            
            loss = self.squared_error(y[i], output_activated) # calculate loss
            self.history['test'].append(float(np.sum(loss))) # add loss computed to 'history'
            
            if len(output_activated[0]) == 1: # fix dimensions of the output
                output_activated = np.sum(output_activated)

            predictions.append(output_activated)

        return predictions


class Layer:
    '''
    Instantiates a Layer which can be used to create the neural network in the Model class.

    Features layers with 4 different activation functions:

    > sigmoid function
        - set activation to 'sigmoid'
    > tanh function
        - set activation to 'tanh'
    > linear function 
        - set activation to 'linear'
    > leaky ReLu
        - set activation to 'leaky_relu'
    > ReLu
        - set activation to 'relu'

    Using an activation function other than these will cause a ValueError.

    :param activation: str - set to 'linear' by default. Other activation functions include: tanh, sigmoid, ReLu and Leaky ReLu.
    :param input_shape: int - input shape of the input layer should match the dimension of the input. For other layers, input shape should match the x_out shape of previous layer.
    :param output_shape: int - x_out shape of the x_out layer should match the dimension of the x_out. For the layers, the x_out shape should match the input shape of the next layer.
    :param seed: int - seed value for the np.random.seed() used to ensure reproducibility; has a default value of 42.
    
    
    *Equations for derivates and functions are discussed in the report
    '''
    def __init__(self, activation="linear", input_shape=1, output_shape=1):
        '''
        Initialise a layer with attributes; activation, input_shape and output_shape
        '''
        self.activation = activation
        self.weights = np.random.randn(input_shape, output_shape) * np.sqrt(2 / (input_shape + output_shape)) # Xavier Glorot Initialisation
        self.bias = np.zeros((1, output_shape))  # set to 0s
        # for adam 
        self.m_W, self.m_B = 0, 0
        self.v_W, self.v_B = 0, 0
        # for sgd + momentum
        self.velocity = np.zeros_like(self.weights)
        self.bias_velocity = np.zeros_like(self.bias) 
        self.alpha = 0.1 # Leaky ReLu gradient for negative inputs.


    def get_activation(self, x):
        '''
        Calculates the x_out using the layer's corresponding activation functions.

        Parameters:
        > x: the x_out value(s) of type float
        > alpha (optional): used as the gradient for leaky ReLu

        Returns:
        > the x_out after activation for that layer as float(s)
        '''
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        
        elif self.activation == "relu":
            return np.maximum(np.zeros_like(x), x)
        
        elif self.activation == "linear":
            return x
        
        elif self.activation == "leaky_relu":
            return np.where(x > np.zeros_like(x), x, self.alpha * x)
        
        elif self.activation == "tanh":
            return np.tanh(x)

        raise ValueError("Invalid Activation ")


    def get_derivative(self, x):
        '''
        Calculates the derivative of the layer's corresponding activation functions.
        Used in backpropagation for training the model.

        Parameters:
        > x: the x_out value(s) before activation of the layer as type float.
        > alpha (optional): used as the gradient for leaky ReLu

        Returns:
        > float or array: The derivative with respect to the pre-activation values for the layer.
        '''
        if self.activation == "sigmoid":
            return self.get_activation(x) * (1 - self.get_activation(x))
        
        elif self.activation == "relu":
            return np.where(x > np.zeros_like(x), 1, 0)
        
        elif self.activation == "linear":
            return 1
        
        elif self.activation == "leaky_relu":
            return np.where(x > np.zeros_like(x), 1, self.alpha)
        
        elif self.activation == "tanh":
            return 1.0 - np.tanh(x)**2
        
        

        

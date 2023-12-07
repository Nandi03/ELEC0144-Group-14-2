from network import Model, Layer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# set seed for reproducability
np.random.seed(42)

# this section allows to modify all parameters of the investigation. Except for singular exception, the rest of the code does not require any editing
# Task 1b - SGD with the default 1-3-1 network.
# Task 1c - SGD with different number of nodes in the hidden layer.
# Task 1d - SGD with different activation functions on each layer.
# Task 1e - Test with different optimisers: Adam, SGD_momentum, SGD_adaptive
# Task 1f - SGD with different training data set sizes 

task = "e"  # select task listed above

nodes = 3 # select number of nodes in the hidden layer

iterations = 100 # select number of iterations, how many times you train the model with same parameters (sample size)

epochs = [300,1000,3000,10000] # select number of epochs to test for

learning_rate = [0.0001,0.00001] # select learning rates to test for

optimiser = ["adam"] # select optimisers to test with

momentum = [0.5,0.8,0.9,0.95,0.99] # test with different momentum values, specifically for sgd + momentum

betas = [[0.9,0.99],[0.9,0.999],[0.99,0.99],[0.99,0.999]] # test with different beta values, specifically for adam

activation = ["tanh"] # select different activation functions 

# modify size of the training set if task 1f is chosen
if task == "f":
    snip = 0.2 # test with 0.2, and 0.5 (20% and 50% respectively taken off the training set)
    x_main = np.arange(-1, 1, 0.05)
    num_elements = int(snip * len(x_main))
else:
    # otherwise, generate the training data set as normal
    x_train = np.arange(-1, 1, 0.05)
    d_train = 0.8 * x_train**3 + 0.3 * x_train**2 - 0.4 * x_train + np.random.normal(0, 0.02, len(x_train)) # adding randomness

# generate the testing data set
x_test = np.arange(-0.97, 0.93, 0.1)
y_test = 0.8* x_test**3 + 0.3 * x_test**2 - 0.4*x_test

# for every learning rate selected in the list
for n in learning_rate:
    print("n = ", n)
    # for every number of epochs selected in the list
    for e in epochs:
        print("e = ", e)
        # for every optimiser selected in the list
        for o in optimiser:
            print("o = ", o)
            # for every activation functon selected in the list
            for a in activation:
                # set paramets for the model
                print("a", a)
                if o == "sgd_momentum":
                    hyperparameters = momentum
                elif o == "adam":
                    hyperparameters = betas
                else:
                    hyperparameters = [1]
                for h in hyperparameters:
                    # curate name of the folder to save images generated and graphs plotted, for analysis
                    print("h = ", h)
                    if task == "c":
                        output_folder = f"1{task}plots_{o}_{a}_{nodes}_n_{str(n).replace('.', '_')}_{e}"
                    if task == "e":
                        if o == 'adam':
                            output_folder = f"1{task}plots_{o}_{a}_n_{str(n).replace('.', '_')}_h_{h[0]}_{h[1]}_{e}"
                        else:
                            output_folder = f"1{task}plots_{o}_{a}_n_{str(n).replace('.', '_')}_h_{h}_{e}"
                    elif task == "f":
                        output_folder = f"1{task}plots_{o}_{a}_n_{str(n).replace('.', '_')}_{snip}_{e}"
                    else:
                        output_folder = f"1{task}plots_{o}_{a}_n_{str(n).replace('.', '_')}_{e}"
                    os.makedirs(output_folder, exist_ok=True)                  

                    # colors for each line graph for every iteration
                    grayscale_values = np.linspace(0, 0.9, iterations)
                    mpl.rcParams['font.size'] = 24
                    plt.figure(figsize=(14,10))

                    # plot graph of the actual testing data
                    plt.scatter(x_test, y_test, color='blue', label='Test Data')

                    # train the model, for 'iterations' number of times
                    for i in range(iterations):
                        
                        if task == "f":
                            selected_indices = np.random.choice(len(x_main), num_elements, replace=False)
                            x_train = x_main[selected_indices]
                            d_train = 0.8 * x_train**3 + 0.3 * x_train**2 - 0.4 * x_train + np.random.normal(0, 0.02, len(x_train))

                        model = Model(epochs = e, learning_rate = n, optimizer = o)
                        if o == "sgd_momentum":
                            model.momentum = h
                        elif o == "adam":
                            model.betas1 = h[0]
                            model.betas2 = h[1]

                        # testing for different activation functions on the hidden layer.
                        model.layers.append(Layer(a, 1, nodes))
                        model.layers.append(Layer("linear", nodes, 1))

                        # variant for 1d when investigating different activation for 
                        # uncomment below for testing different activation function on the output layer
                        #model.layers.append(Layer("tanh", 1, nodes)) 
                        #model.layers.append(Layer(a, nodes, 1))  

                        # train the model
                        model.compile(x_train, d_train)

                        # make predictions using model after training the model
                        predictions = model.fit(x_test,y_test)

                        # PLOTTING GRAPHS
                        # for every iteration, set the color of the line graph
                        color = (grayscale_values[i], grayscale_values[i], grayscale_values[i])
                        # plot the results obtained
                        # showing the x_test values on the x-axis and predicted values on the y-axis
                        plt.plot(x_test, predictions, color=color, label=f'Iteration {i+1}' if i in {0, iterations-1} else None)

                        plt.figure(figsize=(14,10))
                        plt.scatter(x_test, y_test, color='blue', label='Training Data')
                        plt.plot(x_test, predictions, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=8, label='Predictions')
                        plt.xlabel('Unseen data inputs')
                        plt.ylabel('Outputs')
                        plt.title(f'Cubic regression for 1-{nodes}-1 {a} {o} NN ({i+1})')
                        plt.legend()
                        plt.grid(True)
                        prediction_filename = os.path.join(output_folder, f'Prediction_{i+1}.png')
                        plt.savefig(prediction_filename)
                        plt.close() 
                        
                        # if number of epochs is too large, plot the training loss curve
                        # using logarithmic scale
                        if e >= 10000:
                            plt.figure(figsize=(14, 10))
                            plt.plot(model.history['train'], color='blue', label='Cost')
                            plt.yscale('log')
                            plt.xlabel('Epochs')
                            plt.ylabel('Mean Squared Error')
                            plt.title(f'Cost Curve ({i+1})')
                            plt.legend()
                            plt.grid(True)
                            model_history_filename = os.path.join(output_folder, f'ModelHistory_{i+1}.png')
                            plt.savefig(model_history_filename)
                            plt.close()

                        print(i, end=' ')

                    plt.xlabel('Unseen data inputs')
                    plt.ylabel('Outputs')
                    plt.title(f'Cubic regression for 1-{nodes}-1 {a} {o} NN (All)')

                    plt.legend()
                    plt.grid(True)
                    plt.savefig(os.path.join(output_folder, 'AllPredictions.png'))
                    plt.close()

# Confirmation of completion
print("Done")
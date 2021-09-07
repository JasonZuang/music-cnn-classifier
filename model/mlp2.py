import numpy as np
from random import random


class MLP(object):
    """A Multilayer Perceptron class.
    """

    def __init__(self, numIn=3, numHid=[3,4],numOut=2):
        self.numIn = numIn
        self.numHid = numHid
        self.numOut = numOut

        layers = [self.numIn] + self.numHid + [self.numOut]

        self.weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i],layers[i+1])
            self.weights.append(w)

        activations = []
        for i in range (len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range (len(layers)-1):
            a = np.zeros((layers[i],layers[i+1]))
            derivatives.append(a)
        self.derivatives = derivatives


    def forward_propagate(self, input):
        activation = input
        self.activations[0] = input
        for i, w in enumerate(self.weights):
            #net inputs
            netIn = np.dot(activation,w)
            
            #calc activations

            activation = self._sigmoid(netIn)
            self.activations[i+1] = activation
        return activation

    def back_propagate(self, error, verbose = False):
        '''
            dE/dW_i = (y - a[i+1]) s'(h[i+1]) a
            s'(h[i+1]) = s(h[i+1])(1 - s(h[i+1_]))
            s(h[i+1]) = a[i+1]
            dE/dW = de/dw * w_i s'(h_i)a[i-1]
        '''
        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations)
            delta_re = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations = current_activations.reshape(current_activations.shape[0],-1)
            self.derivatives[i] = np.dot(current_activations, delta_re)
            error = np.dot(delta, self.weights[i].T)


    def train(self, x, y, e, a):
        sum_errors = 0
        for i in range(e):  

        # iterate through all the training data
            for j, input in enumerate(x):
                y = y[j]

        # activate the network!
                output = self.forward_propagate(input)

                error = y - output

                self.back_propagate(error)

        # now perform gradient descent on the derivatives
        # (this will update the weights
                self.gradient_descent(a)

        # keep track of the MSE for reporting later
                sum_errors += self._mse(y, output)

        # Epoch complete, report the training error
                print("Error: {} at epoch {}".format(sum_errors / len(items), i+1))


    def gradient_descent(self, a):
        for i in range(len(self.weights)):
            
            weights = self.weights[i]
            d = self.derivatives[i]
            weights += d*a


    def _sigmoid(self, x):
        """Sigmoid activation function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """

        y = 1.0 / (1 + np.exp(-x))
        return y


    def _sigmoid_derivative(self, x):
        """Sigmoid derivative function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """
        return x * (1.0 - x)


    def _mse(self, target, output):
        """Mean Squared Error loss function
        Args:
            target (ndarray): The ground trut
            output (ndarray): The predicted values
        Returns:
            (float): Output
        """
        return np.average((target - output) ** 2)


if __name__ == "__main__":

    # create a dataset to train a network for the sum operation
    items = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])

    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(2, [5], 1)

    # train network
    mlp.train(items, targets, 50, 0.1)

    # create dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    # get a prediction
    output = mlp.forward_propagate(input)

    print()
    print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))


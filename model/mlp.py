import numpy as np
from random import random
# save activations and derivatives
# implement backpropagation
# implement gradient descent
# implement train descent

class MLP:
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

	def forwardProp(self, input):
		activation = input
		self.activations[0] = input
		for i, w in enumerate(self.weights):
			#net inputs
			netIn = np.dot(activation,w)
			
			#calc activations

			activation = self._sigmoid(netIn)
			self.activations[i+1] = activation
		return activation
	def backProp(self, error, verbose = False):
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

	def grad_des(self, a):
		for i in range(len(self.weights)):
			
			weights = self.weights[i]
			print("weights ",weights)
			d = self.derivatives[i]
			weights += d*a
			print("weights ",weights)

	def train(self, x, y, e, a):
		
		for i in range(e):
			sum_errors = 0
		# iterate through all the training data
			for j, input in enumerate(x):
				y= y[j]

		# activate the network!
				output = self.forwardProp(input)

				error = y - output

				self.backProp(error)

		# now perform gradient descent on the derivatives
		# (this will update the weights
				self.gradient_descent(a)

		# keep track of the MSE for reporting later
				sum_errors += self._mse(y, output)

		# Epoch complete, report the training error
				print("Error: {} at epoch {}".format(sum_errors / len(items), i+1))

	def _mse(x,o):
		return np.average((y-o)**2)

	def _sigmoid_derivative(self,h):
		return h*(1-h)

	def _sigmoid(self,h):
		return (1/(1+np.exp(-h)))


if __name__ == "__main__":

	mlp = MLP(2,[5],1)

	# dummy data
	x = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
	y = np.array([[i[0] + i[1]] for i in x])

	x_1 = np.array([.1,.1])
	y_1 = np.array([.2])

	mlp.train(x_1,y_1,1,0.1)


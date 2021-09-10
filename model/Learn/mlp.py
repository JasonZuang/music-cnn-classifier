import numpy as np 
from random import random

class MLP(object):

	def __init__(self, numIn = 2, layers = [5], numOut = 1):

		self.numIn = numIn
		self.layers = layers
		self.numOut = numOut

		layers = [numIn] + layers + [numOut]

		weights = []
		for i in range(len(layers)-1):
			w = np.random.rand(layers[i],layers[i+1])
			weights.append(w)
		self.weights = weights

		derivatives = []
		for i in range(len(layers)-1):
			d = np.zeros((layers[i],layers[i-1]))
			derivatives.append(d)
		self.derivatives = derivatives

		activations = []
		for i in range(len(layers)):
			a = np.zeros(layers[i])
			activations.append(a)
		self.activations = activations

	def forProp(self, inputs):
		activations = inputs
		self.activations[0] = activations
		for i,w in enumerate(self.weights):
			dotA = np.dot(activations,w)
			activations = self.sigmoid(dotA)
			self.activations[i+1] = activations
		return activations

	def backProp(self,e):
		for i in reversed(range(len(self.derivatives))):
			activations = self.activations[i+1]
			d = e*self.sigmoid_der(activations)
			dR = d.reshape(d.shape[0],-1).T
			currActivation = self.activations[i]
			currActR = currActivation.reshape(currActivation.shape[0],-1)
			self.derivatives[i] = np.dot(currActR,dR)
			e = np.dot(d,self.weights[i].T)
		return e

	def train(self, _x, _y, e, a):
		for i in range(e):
			for j, x in enumerate(_x):
				y = _y[j]
				o = self.forProp(x)
				error = y-o
				self.backProp(error)
				self.gradDes(a)

	def gradDes(self, a):
		for i in range(len(self.weights)):
			weights = self.weights[i]
			der = self.derivatives[i]
			weights += der * a

	def sigmoid(self,i):
		return (1/(1+np.exp(-i)))
	
	def sigmoid_der(self,i):
		return i * (1-i)

if __name__ == "__main__":

	# Model is trained to multiply two numbers
	mlp = MLP(2,[5],1)
	x = np.array([[random()/2 for _ in range(2)] for _ in range(10000)])
	y = np.array([[i[0] * i[1]] for i in x])
	
	mlp.train(x,y,50,0.1)

	x_test = np.array([[.1,.2]])
	y_test = np.array([[.02]])
	print(mlp.forProp(x_test)," Expected output :", y_test)

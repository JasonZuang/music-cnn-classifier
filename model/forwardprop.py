import numpy as np

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

	def forwardProp(self, input):
		activation = inputs
		for w in self.weights :
			#net inputs
			netIn = np.dot(activation,w)
			
			#calc activations

			activation = self._sigmoid(netIn)

		return activation

	def _sigmoid(self,h):
		return (1/(1+np.exp(-h)))


if __name__ == "__main__":

	numIn = 3
	numHid = [3,4,5]
	numOut = 2

	mlp = MLP(numIn,numHid,numOut)

	inputs = np.random.rand(mlp.numIn)

	outputs = mlp.forwardProp(inputs)

	print(outputs)

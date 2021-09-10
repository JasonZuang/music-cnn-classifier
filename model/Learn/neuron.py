import math

def sigmoid(h):
	return (1/(1 + math.exp(-h)))
	
def activate(inputs, weights):
	h = 0
	for x,w in zip(inputs,weights):
		h += x*w
	return sigmoid(h)

if __name__ == "__main__":
	inputs = [.5,.3,.2]
	weights = [.4,.7,.2]
	output = activate(inputs,weights)
	print(output)



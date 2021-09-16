import numpy as np
from random import random
from sklearn.model_selection import train_test_split
import tensorflow as tf

'''
Using Tensorflow to
BUILD
COMPILE
TRAIN
EVAL
PREDICT
'''
#creating test sets
def genData(numSam, testSize):
	x = np.array([[random()/2 for _ in range(2)] for _ in range(numSam)])
	y = np.array([[i[0] + i[1]] for i in x])
	xTrain, xTest, yTrain,yTest = train_test_split(x,y,test_size = testSize)
	return xTrain, xTest, yTrain, yTest


if __name__ == "__main__":
	xTr,xT, yTr,yT = genData(5000,0.3)


	#build model 2 -> 5 -> 5 -> 1
	model = tf.keras.Sequential([
		tf.keras.layers.Dense(5,input_dim = 2, activation="sigmoid"),
		tf.keras.layers.Dense(1,activation = "sigmoid")
	])

	#compile model
	o = tf.keras.optimizers.SGD(learning_rate=0.1)
	model.compile(optimizer=o,loss="MSE")

	#train model
	model.fit(xTr,yTr,epochs=100)

	#eval model
	model.evaluate(xT,yT, verbose = True)

	#predict
	xP = np.array([[0.4,0.5]])
	yP = np.array([[0.8]])

	prediction = model.predict(xP)

	print(prediction, yP)

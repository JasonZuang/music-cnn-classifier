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
	y = np.array([[i[0] * i[1]] for i in x])
	xTrain, xTest, yTrain,yTest = train_test_split(x,y,test_size = testSize)
	return xTrain, xTest, yTrain, yTest


if __name__ == "__main__":
	xTr,xT, yTr,yT = genData(100,0.3)


	#build model 2 -> 6 -> 1
	model = tf.keras.Sequential([
		tf.keras.layers.Dense(5,input_dim = 2, activation="sigmoid"),
		tf.keras.layers.Dense(1,activation = "sigmoid")
	])

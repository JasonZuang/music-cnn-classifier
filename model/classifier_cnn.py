import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATAPATH = "Data/data.json"
TESTDATA = "Data/data_test.json"

def load(dataPath):
	with open(dataPath,"r") as fp:
		data = json.load(fp)

	inputs = np.array(data['mfcc'])
	targets = np.array(data['labels'])
	genre = np.array(data['mapping'])
	return inputs,targets,genre



def prepData(inputs,targets, testSize, valSize):
	xTrain, xTest, yTrain, yTest = train_test_split(inputs,
													targets,
													test_size = testSize)
	

	#creating validation set
	xTrain, xVal, yTrain, yVal = train_test_split(xTrain,
												  yTrain,
												  test_size = valSize)

	#creating 3d 
	xTrain = xTrain[...,np.newaxis]
	xVal = xVal[...,np.newaxis]
	xTest = xTest[...,np.newaxis]

	return xTrain, xVal, xTest, yTrain, yVal, yTest


def buildModel(inputShape):
	model = keras.Sequential()

	model.add(keras.layers.Conv2D(32, (3,3), activation = "relu", input_shape = inputShape))

	model.add(keras.layers.MaxPool2D((3,3),strides =(2,2), padding="same"))

	#normalizes activations in the normal layers and speed up the training
	model.add(keras.layers.BatchNormalization())

	model.add(keras.layers.Conv2D(32, (3,3), activation = "relu", input_shape = inputShape))

	model.add(keras.layers.MaxPool2D((3,3),strides =(2,2), padding="same"))

	#normalizes activations in the normal layers and speed up the training
	model.add(keras.layers.BatchNormalization())

	model.add(keras.layers.Conv2D(32, (2,2), activation = "relu", input_shape = inputShape))

	model.add(keras.layers.MaxPool2D((2,2),strides =(2,2), padding="same"))

	#normalizes activations in the normal layers and speed up the training
	model.add(keras.layers.BatchNormalization())

	model.add(keras.layers.Flatten())

	model.add(keras.layers.Dense(64,activation ="relu"))

	model.add(keras.layers.Dropout(0.3))

	model.add(keras.layers.Dense(10, activation="softmax"))

	return model

def plotHistory(history):

	fig , axis = plt.subplots(2)
	axis[0].plot(history.history["accuracy"],label = "train accuracy")
	axis[0].plot(history.history["val_accuracy"], label = "test accuracy")
	axis[0].set_ylabel("accuracy")
	axis[0].legend(loc="lower right")
	axis[0].set_title("Accuracy Eval")

	axis[1].plot(history.history["loss"],label = "train loss")
	axis[1].plot(history.history["val_loss"], label = "test loss")
	axis[1].set_ylabel("loss")
	axis[1].set_xlabel("epoch")
	axis[1].legend(loc="lower right")
	axis[1].set_title("loss Eval")

	plt.show()

def predict(model,x,y, genre):
	x = x[np.newaxis,...]
	prediction = model.predict(x) 
	predictedIndex = np.argmax(prediction, axis = 1) # makes [3]
	print("Prediction", genre[predictedIndex], " Actual ", genre[y])
if __name__ == "__main__":

	#create train, validation and test set
	inputs, targets, genre = load(DATAPATH)

	xTrain, xVal, xTest, yTrain, yVal, yTest = prepData(inputs, targets,0.25, 0.2)

	#build the CNN net
	inputShape = (xTrain.shape[1], xTrain.shape[2], xTrain.shape[3])
	model = buildModel(inputShape)


	#compile network
	optimizer = keras.optimizers.Adam(learning_rate =0.0001)
	model.compile(optimizer = optimizer,
				  loss = "sparse_categorical_crossentropy",
				  metrics = ["accuracy"])

	#train CNN 

	history = model.fit(xTrain,yTrain,validation_data = (xVal,yVal), batch_size = 32, epochs = 100)
	

	#eval CNN on test
	print("*************************")
	testE, testA = model.evaluate(xTest,yTest, verbose = 0)
	print("Accuracy", testA)


	#predict
	with open(TESTDATA,"r") as fp:
		data = json.load(fp)

		x = np.array(data['mfcc'][2])
		x = x[...,np.newaxis]
		y = 5
		print(x.shape)
		predict(model,x,y, genre)
		predict(model,xTest[2],yTest[2],genre)
		predict(model,xTest[10],yTest[10],genre)

	model.save('music_classifier')



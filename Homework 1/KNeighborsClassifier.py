import pandas as pd
import random
import math
import operator
import matplotlib.pyplot as plt

def loadDataset(url, split):
	trainingSet=[]
	testSet=[]
	df = pd.read_csv(url, header=None)
	array = df.to_numpy()
	random.shuffle(array)
	training_len = int(len(array)*split)
	trainingSet = array[:training_len]
	testSet = array[training_len:]
	return trainingSet, testSet


def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	
def main():
	# prepare data
	ypoints=[]
	xpoints=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
	trainingSet=[]
	testSet=[]
	split = 0.67
	fileName = "homework 1/iris.data"
	print('Train set: ' + repr(len(trainingSet)))
	print('Test set: ' + repr(len(testSet)))
	# generate predictions
	predictions=[]
	k = 1
	while (k <= 20):
		print('K: ' + str(k))
		k = k + 1
		trainingSet, testSet = loadDataset(fileName, split)

		# loop through testSet
		for x in range(len(testSet)):
			# TODO starts here
			# get neighbor between current test record and all training datasets
			neighbors = getNeighbors(trainingSet, testSet[x], k)
			# get response
			result = getResponse(neighbors)
			# append current prediction result to predictions list
			predictions.append(result)
			#print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
			# TODO ends here
		accuracy = getAccuracy(testSet, predictions)
		print('Accuracy: ' + repr(accuracy) + '%')
		ypoints.append(repr(accuracy))
	plt.title("Iris Flower KNN")
	plt.xlabel("K Value")
	plt.ylabel("Percent Accuracy")
	plt.plot(xpoints, ypoints)
	plt.show()
main()

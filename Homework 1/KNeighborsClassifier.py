import pandas as pd
import random

def loadDataset(filename, split):
	trainingSet=[]
	testSet=[]
	df = pd.read_csv(filename, header=None)
	array = df.to_numpy()
	random.shuffle(array)
	training_len = int(len(array)*split)
	trainingSet = array[:training_len]
	testSet = array[training_len:]
	return trainingSet, testSet

def main():
    filename = 'homework 1/iris.data'
    trainingSet, testSet = loadDataset(filename, 0.66)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))

main()
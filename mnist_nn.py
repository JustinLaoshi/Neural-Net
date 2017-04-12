from __future__ import division
from collections import Counter
from NeuralNetwork import NeuralNetwork
import numpy as np
import scipy.io
import pdb
import math
import random
import sklearn
from sklearn.feature_extraction import DictVectorizer
import itertools
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt 


def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = np.sum(errors) / float(true_labels.shape[0])
    indices = errors.nonzero()
    return err_rate, indices

def main():
	runNN()

def runNN():
	#load all the data
	train = scipy.io.loadmat('dataset/train.mat')
	test = scipy.io.loadmat("dataset/test.mat")
	train_labels = train['train_labels']
	train_images = train['train_images'].transpose(2, 0, 1).reshape(60000, -1)
	train_labels, train_images = sklearn.utils.shuffle(train_labels, train_images)
	#train_images = sklearn.preprocessing.normalize(train_images)
	test_images = test['test_images']
	test_images = test_images.reshape(10000, 784)
	#test_images = sklearn.preprocessing.normalize(test_images)


	# for i in range(0, 25):
	# 	plt.imshow(test_images[i].reshape(28, 28))
	# 	plt.show()	


	def validate(train_images, train_labels, bias=False, stoch=True, MSE=True, pickle=False):
		"""
		Prints the error rate of verifying on a validation set of 10,000 images set aside.
		"""
		if bias:
			train_images = np.insert(train_images, 784, 1.0, axis=1)
			nn = NeuralNetwork(784, 10, 400, bias, pickle)
		else:
			nn = NeuralNetwork(784, 10, 400, bias, pickle)
		tempTrain, tempLabels = train_images[:50000], train_labels[:50000]
		valid_images, valid_labels = train_images[50000:], train_labels[50000:]
		
		nn.train(tempTrain, tempLabels, 0.01, stoch, MSE)
		pred_labels = nn.predict(tempTrain)
		#makeAndShowPlotsCost(nn)
		print(benchmark(pred_labels, tempLabels)[0])

	def predictTest(test_images, train_images, train_labels, bias=False, MSE=True):
		"""
		Predict on the test set of images.
		"""
		if bias:
			train_images = np.insert(train_images, 784, 1.0, axis=1)
			test_images = np.insert(test_images, 784, 1.0, axis=1)
			nn = NeuralNetwork(784, 10, 400, bias)
		else:
			nn = NeuralNetwork(784, 10, 400, bias)
		nn.train(train_images, train_labels, 0.01, True, MSE)
		pred_labels = nn.predict(test_images)
		return pred_labels

	def makeCSV(pred_labels):
		"""
		Makes the CSV for Kaggle.
		"""
		csvList = [['Id,Category']]
		for i in range(1, 10001):
		    csvList.append([i, int(pred_labels[i-1][0])])
		with open('nn.csv', 'w', newline='') as fp:
		    a = csv.writer(fp, delimiter=',')
		    a.writerows(csvList)
		return 0	

	def makeAndShowPlotsCost(nn):
		yAxis = nn.errorList
		xAxis = nn.iterList
		plt.plot(xAxis, yAxis, 'ro')
		plt.ylabel('Cost')
		plt.xlabel('Number of Iterations')
		axes = plt.gca()
		plt.show()		
	
	def makeAndShowPlotsClassAcc():
		xAxis = [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000]
		#first yaxis is cross entropy, second is mean squared
		yAxis = [0, .87,      .9,   .92,     .95,    .96,    .97,    .98,    .982,   .983,  .984 ]
		#yAxis = [0, .87,      .9,   .92,     .94,    .95,    .96,    .97,    .972,   .973,  .974 ]
		plt.plot(xAxis, yAxis, 'ro')
		plt.ylabel('Validation Accuracy')
		plt.xlabel('Number of Iterations')
		axes = plt.gca()
		plt.show()	


	#makeAndShowPlotsClassAcc()
	#validate(train_images, train_labels, True, True, False, True)

	#pred_labels = predictTest(test_images, train_images, train_labels, True, False)
	#makeCSV(pred_labels)









if __name__ == "__main__":
    main()

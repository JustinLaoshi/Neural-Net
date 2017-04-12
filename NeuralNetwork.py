from __future__ import division
from collections import Counter
import numpy as np
import scipy.io
import pdb
import math
import random
import pickle
import sklearn
from sklearn.feature_extraction import DictVectorizer
import itertools
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt 

class NeuralNetwork(object):

	def __init__(self, inputLayerSize, outputLayerSize, hiddenLayerSize, bias, p=False):
		"""
		Initializes the Neural Network and the weights.
		"""
		self.inputLayerSize = inputLayerSize
		self.hiddenLayerSize = hiddenLayerSize
		self.outputLayerSize = outputLayerSize
		self.bias = bias
		if bias:
			if p:
				print('gets pickle')
				self.V = pickle.load(open('V.p', 'rb'))
				self.W = pickle.load(open('W.p', 'rb'))
			else:
				self.V = 0.01 * np.random.randn(self.inputLayerSize + 1, self.hiddenLayerSize)
				self.W = 0.01 * np.random.randn(self.hiddenLayerSize + 1, self.outputLayerSize)
		else:
			if p:
				print('gets pickle without bias')
				self.V = pickle.load(open('V.p', 'rb'))
				self.W = pickle.load(open('W.p', 'rb'))			
			else:

				self.V = 0.01 * np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
				self.W = 0.01 * np.random.randn(self.hiddenLayerSize, self.outputLayerSize)





	def sigmoid(self, z):
		if z.all() >= 0:
			z = np.exp(-z)
			return 1/(1+z)
		else:
			z = np.exp(z)
			return z/(1+z)

	def sigmoidPrime(self, z):
		return self.sigmoid(z) * (1 - self.sigmoid(z))

	def tanh(self, x):
		return np.tanh(x)

	def tanhPrime(self, x):
		return 1 - (np.tanh(x))**2

	def oneOfNOutEncoding(self, value):
		vec = np.zeros((1, 10))

		#vec.fill(0.15)
		vec[0][value] = 1
		return vec




	def crossEntropyError(self, x, y, stoch=True):
		self.yHat = self.forward(x, stoch)
		value = y[0]
		vec = self.oneOfNOutEncoding(value)
		one = np.ones((10, 1))
		J = -sum(vec.T*np.log(self.yHat.T) + (one-self.yHat.T)*np.log(one - self.yHat.T))
		return J

	def crossEntropyErrorPrime(self, x, y, stoch=True):
		self.yHat = self.forward(x, stoch)
		if stoch:
			if self.bias:
				x.shape = (1, 785)
			else:
				x.shape = (1, 784)
			y = self.oneOfNOutEncoding(y)
		one = np.ones((1, 10))
		term = np.divide(-y, self.yHat) + np.divide(one-y, one-self.yHat)
		delta3 = np.multiply(term, self.sigmoidPrime(self.z3))
		if self.bias:
			dJdW = np.dot(self.a2Bias.T, delta3)
		else:
			dJdW = np.dot(self.a2.T, delta3)
		
		if self.bias:
			self.z2Bias = np.insert(self.z2, self.hiddenLayerSize, 1.0, axis=1)	
			delta2 = np.multiply(np.dot(delta3, self.W.T), self.tanhPrime(self.z2Bias))
		else:			
			delta2 = np.multiply(np.dot(delta3, self.W.T), self.tanhPrime(self.z2))		
		dJdV = np.dot(x.T, delta2)
		if self.bias:
			dJdV = np.delete(dJdV, self.hiddenLayerSize, 1)		
		return dJdV, dJdW

	def meanSquaredErrorPrime(self, x, y, stoch=True):
		self.yHat = self.forward(x, stoch)
		if stoch:
			if self.bias:
				x.shape = (1, 785)
			else:
				x.shape = (1, 784)		
			y = self.oneOfNOutEncoding(y)

		delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
		if self.bias:
			dJdW = np.dot(self.a2Bias.T, delta3)
		else:
			dJdW = np.dot(self.a2.T, delta3)

		if self.bias:
			self.z2Bias = np.insert(self.z2, self.hiddenLayerSize, 1.0, axis=1)	
			delta2 = np.multiply(np.dot(delta3, self.W.T), self.tanhPrime(self.z2Bias))
		else:			
			delta2 = np.multiply(np.dot(delta3, self.W.T), self.tanhPrime(self.z2))
	#	if self.bias:
		dJdV = np.dot(x.T, delta2)
		if self.bias:
			dJdV = np.delete(dJdV, self.hiddenLayerSize, 1)
		return dJdV, dJdW


	def meanSquaredError(self, x, y, stoch=True):
		self.yHat = self.forward(x, stoch)
		
		value = y[0]
		vec = self.oneOfNOutEncoding(value)
		J = 0.5 * sum((vec.T - self.yHat.T)**2)
		return J

	def forward(self, x, stoch=True):
		if self.bias:
			x.shape = (1, 785)
		else:
			x.shape = (1, 784)

		self.z2 = np.dot(x, self.V)
		if stoch == True:
			self.z2.shape = (1, self.hiddenLayerSize) 

		self.a2 = self.tanh(self.z2)
		if self.bias:
			self.a2Bias = np.insert(self.a2, self.hiddenLayerSize, 1.0, axis=1)	
			self.z3 = np.dot(self.a2Bias, self.W)
		else:
			self.z3 = np.dot(self.a2, self.W)
		yHat = self.sigmoid(self.z3)
		return yHat







	def train(self, X, y, e, stoch=True, MSE=True):
		self.errorList = []
		self.iterList = []
		for i in range(0, 50000):

			print('Now at iteration:', i)
			index = random.randint(0, X.shape[0]-1)
			temp = np.divide(X[index], 255)
			if stoch:
				if not MSE:
					dJdV, dJdW = self.crossEntropyErrorPrime(temp, y[index], stoch)
				else:
					dJdV, dJdW = self.meanSquaredErrorPrime(temp, y[index], stoch)
			else:
				if not MSE:
					dJdV, dJdW = self.crossEntropyErrorPrime(X, y, stoch)
				else:
					dJdV, dJdW = self.meanSquaredErrorPrime(X, y, stoch)
			self.V = self.V - e * dJdV
			self.W = self.W - e * dJdW
			if MSE:
				print('current error:', self.meanSquaredError(temp, y[index]))
				if i % 10000 == 0:
					self.iterList.append(i)	
					self.errorList.append(self.meanSquaredError(temp, y[index]))
			else:
				print('current error:', self.crossEntropyError(temp, y[index]))
				if i % 10000 == 0:
					self.iterList.append(i)	
					self.errorList.append(self.crossEntropyError(temp, y[index]))		
					
		# Saving the objects:
		tempV = self.V
		tempW = self.W
		pickle.dump(tempV, open('V.p', 'wb'))
		pickle.dump(tempW, open('W.p', 'wb'))



	def predict(self, test):
		result = np.zeros((test.shape[0], 1))
		i = 0
		for x in test:
			self.yHat = self.forward(x, True)

			result[i] = np.argmax(self.yHat, axis=1)[0]
			i += 1


		return result







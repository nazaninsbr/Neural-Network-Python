from activationFunctions import *
import numpy as np

class Neuron:
	def __init__(self, pixel_values, weights):
		self.pixel_values = np.asarray(pixel_values)
		self.weights = np.asarray(weights)
		self.sigma = 0.0
		self.output = 0
		self.dropOut = 0

	def calc_sigma(self):
		self.sigma = np.dot(self.pixel_values, self.weights)

	def calc_activation(self):
		self.output = sigmoid(self.sigma)


	def calc_output(self):
		if self.dropOut==0:
			self.calc_sigma()
			self.calc_activation()
			return self.output
		elif self.dropOut==1:
			return 0

	def updateWeights(self, updatedWeights):
		#self.weights = np.array([])
		np.copyto(self.weights , updatedWeights)
		#print("weightsupdated" , len(self.weights))

	def setNewInput(self, newInput):
		cnt = 0
		self.pixel_values = np.array([])
		self.pixel_values = np.asarray(newInput)
		

	def getWeights(self):
		return self.weights

	def getNumOfWeights(self):
		return self.weights.size

	def setDropOut(self):
		self.dropOut = 1

	def resetDropOut(self):
		self.dropOut = 0
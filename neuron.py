from activationFunctions import *
import numpy as np

class Neuron:
	def __init__(self, pixel_values, weights):
		self.pixel_values = np.asarray(pixel_values)
		self.weights = np.asarray(weights)
		self.sigma = 0
		self.output = 0

	def calc_sigma(self):
		# print(self.pixel_values)
		# print(self.weights)
		# print(type(self.pixel_values))
		self.sigma = np.dot(self.pixel_values, self.weights)

	def calc_activation(self):
		self.output = sigmoid(self.sigma)


	def calc_output(self):
		self.calc_sigma()
		self.calc_activation()
		return self.output

	def setNewInput(self, newInput):
		self.pixel_values = np.asarray(newInput)

	def getWeights(self):
		return self.weights
from activationFunctions import *
import numpy as np

class Neuron:
	def __init__(self, pixel_values, weights):
		self.pixel_values = np.asarray(pixel_values)
		self.weights = np.asarray(weights)
		self.sigma = 0.0
		self.output = 0

	def calc_sigma(self):
		#print("len of pixel_values",self.pixel_values.size)
		#print("len of weights",self.weights.size)
		#print(type(self.pixel_values))
		#if self.pixel_values == 'null':
			#print("NULL")
		#print("dot : " , np.dot(self.pixel_values, self.weights))
		self.sigma = np.dot(self.pixel_values, self.weights)
		#self.sigma = np.dot(np.squeeze(self.pixel_values), self.weights)

	def calc_activation(self):
		self.output = sigmoid(self.sigma)


	def calc_output(self):
		self.calc_sigma()
		self.calc_activation()
		return self.output

	# def updateWeights(self, updatedWeights):
	# 	#TODO: delete one elements 
	# 	self.weights = self.weights[-1]
	# 	self.weights = np.delete(self.weights , 0)
	# 	np.place(self.weights , updatedWeights ,  )

	def setNewInput(self, newInput):
		cnt = 0
		#print ("newInput in Neuron: " ,newInput)
		self.pixel_values = np.array([])
		print("pixel"  , self.pixel_values)
		#print(type(newInput))
		self.pixel_values = np.asarray(newInput)
		#print(len(newInput))
		#print(self.pixel_values.size)
		#print("pixel_values in neuron: "  , self.pixel_values)

	def getWeights(self):
		return self.weights

	def getNumOfWeights(self):
		return self.weights.size

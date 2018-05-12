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

	def setNewInput(self, newInput):
		cnt = 0
		#print ("newInput in Neuron: " ,newInput)

		# 		
		# 
		# TODO:delete elemet
		# 
		# 
		# 		
		# self.pixel_values = self.pixel_values[-1]
		# self.pixel_values = np.delete(self.pixel_values , 0)
		#print(self.pixel_values)

		#print(type(newInput))
		#for i in range (0,newInput.length):
			#cnt +=1
		#print(cnt)
		self.pixel_values = np.asarray(newInput)
		#print(len(newInput))
		#print(self.pixel_values.size)
		#print("pixel_values in neuron: "  , self.pixel_values)

	def getWeights(self):
		return self.weights

	def getNumOfWeights(self):
		return self.weights.size

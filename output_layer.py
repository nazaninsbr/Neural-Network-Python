from neuron import *
import random 
import numpy as np

class OutputLayer:
	def __init__(self, number_of_neurons, input_values):
		self.input_values = input_values
		self.number_of_neurons = number_of_neurons
		self.neurons = []
		self.outputValues = []
		self.work()

	def work(self):
		self.createNeurons()
		self.calcOutputs()

	def createNeurons(self):
		# print("num of out lay neurons: ", self.number_of_neurons)
		for i in range(self.number_of_neurons):
			w = [random.uniform(-2, 2) for x in self.input_values]
			newNeuron = Neuron(self.input_values, w)
			self.neurons.append(newNeuron)

	def calcOutputs(self):
		del self.outputValues[:]
		for n in self.neurons:
			self.outputValues.append(n.calc_output())


	def setWeightsUpdated(self , updatedWeights):
		
		for i in range(0 , len(self.neurons)):
			updatedWeightsTemp = updatedWeights[i]
			self.neurons[i].updateWeights(updatedWeightsTemp)

	def getOutput(self):
		return self.outputValues

	def setNewInput(self,newInput):
		self.input_values = newInput
		for n in self.neurons:
			n.setNewInput(self.input_values)
		self.calcOutputs()
		return self.getOutput()

	def getWeights(self):
		li = []
		for n in self.neurons:
			li.append(n.getWeights())
		return li

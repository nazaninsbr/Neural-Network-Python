from neuron import *
import random 

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
		for i in range(self.number_of_neurons):
			w = [random.uniform(0, 1) for x in self.input_values]
			newNeuron = Neuron(self.input_values, w)
			self.neurons.append(newNeuron)

	def calcOutputs(self):
		for n in self.neurons:
			self.outputValues.append(n.calc_output())

	def getOutput(self):
		return self.outputValues

	def setNewInput(newInput):
		self.input_values = newInput
		for n in self.neurons:
			n.setNewInput(self.input_values)


		return self.calcOutputs()

	def getWeights(self):
		li = []
		for n in self.neurons:
			li.extend(n.getWeights())
		return li

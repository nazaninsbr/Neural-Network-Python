from neuron import *
import random 

class HiddenLayer:
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
			bias = -1
			w = random.uniform(0, 1)
			newNeuron = Neuron(self.input_values, w, bias)
			self.neurons.append(newNeuron)

	def calcOutputs(self):
		for n in self.neurons:
			self.outputValues.append(n.calc_output())

	def getOutput(self):
		return self.outputValues

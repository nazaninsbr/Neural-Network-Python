from neuron import *
import random 
from activationFunctions import *

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
			w = [random.uniform(0, 1) for x in self.input_values]
			newNeuron = Neuron(self.input_values, w)
			self.neurons.append(newNeuron)
			del w[:]

	def calcOutputs(self):
		del self.outputValues[:]
		for n in self.neurons:
			self.outputValues.append(n.calc_output())

	def getOutput(self):
		return self.outputValues


	def setNewInput(self, newInput):
		#print("len of newInput in hiddenLayer",len(newInput))
		self.input_values = newInput
		for n in self.neurons:
			n.setNewInput(self.input_values)


		self.calcOutputs()
		return self.getOutput()


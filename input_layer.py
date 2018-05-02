from neuron import *
import random 

class InputLayer:
	def __init__(self, input_values):
		self.input_values = input_values
		self.neurons = []
		self.outputValues = []
		self.work()

	def work(self):
		self.createNeurons()
		self.calcOutputs()

	def createNeurons(self):
		for i in range(len(self.input_values)):
			for j in range(len(self.input_values[0])):
				bias = -1
				w = random.uniform(0, 1)
				newNeuron = Neuron(self.input_values[i][j], w, bias)
				self.neurons.append(newNeuron)

	def calcOutputs(self):
		for n in self.neurons:
			self.outputValues.append(n.calc_output())

	def getOutput(self):
		return self.outputValues

from neuron import *
import random 

class InputLayer:
	def __init__(self, input_values):
		self.number_of_pixels = 28
		self.input_values = input_values
		self.neurons = []
		self.outputValues = []
		self.work()

	def work(self):
		self.createNeurons()
		self.calcOutputs()

	def createNeurons(self):
		for i in range(self.number_of_pixels):
			for j in range(self.number_of_pixels):
				inputs = [x[i][j] for x in self.input_values]
				w = [random.uniform(0, 1) for x in inputs]
				newNeuron = Neuron(inputs, w)
				self.neurons.append(newNeuron)

	def calcOutputs(self):
		for n in self.neurons:
			self.outputValues.append(n.calc_output())

	def getOutput(self):
		return self.outputValues

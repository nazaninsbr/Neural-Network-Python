from neuron import *
import random 

class InputLayer:
	def __init__(self, input_values, option=1):
		self.number_of_pixels = 28
		self.input_values = input_values
		self.neurons = []
		self.outputValues = []
		self.work(option)

	def work(self, option):
		if option==1:
			self.createNeurons()
			self.calcOutputs()
		elif option==2:
			self.sgdcreateNeurons()
			self.calcOutputs()

	def createNeurons(self):
		for i in range(self.number_of_pixels):
			for j in range(self.number_of_pixels):
				inputs = [x[i][j] for x in self.input_values]
				w = [random.uniform(0, 1) for x in inputs]
				newNeuron = Neuron(inputs, w)
				self.neurons.append(newNeuron)

	def sgdcreateNeurons(self):
		for i in range(self.number_of_pixels):
			for j in range(self.number_of_pixels):
				inputs = [self.input_values[i][j]]
				w = [random.uniform(0, 1)]
				newNeuron = Neuron(inputs, w)
				self.neurons.append(newNeuron)


	def calcOutputs(self):
		del self.outputValues[:]
		for n in self.neurons:
			self.outputValues.append(n.calc_output())

	def getOutput(self):
		return self.outputValues

	def setWeightsUpdated(self , updatedWeights):
		#TODO: testing...
		x = len(self.getWeights())
		updatedWeightsTemp = updatedWeights[:x]
		# if updatedWeights is numpy delete is false
		del updatedWeights[:x]
		for n in self.neurons:
			n.updatedWeights(updatedWeightsTemp)



	def getWeights(self ):
		li = []
		for n in self.neurons:
			li.extend(n.getWeights())
		return np.asarray(li)

	def setNewInput(self, newInput):
		#print ("newInput in input layer: " , newInput)
		self.input_values = newInput
		#print("the len of input_values in input layer -> "  , len(self.input_values))
		i = 0
		j = 0
		for n in self.neurons:
			#print("my len is :"  , len(self.neurons))
			inputval = self.input_values[i][j]
			#print("say I'm i: ",i)
			#print("say I'm j: ",j)
			j +=1
			if(j==28):
				j = 0;
				i = (i + 1)%28
			#print ("inputval in input layer : " , inputval)
			n.setNewInput(inputval)
			#print("inputval in input layer: " , inputval)
		self.calcOutputs()
		#print("size of output of set new out in input layer: "  , len(self.getOutput()))
		return self.getOutput()


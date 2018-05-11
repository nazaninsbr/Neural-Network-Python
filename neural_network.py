from input_layer import *
from hidden_layer import *
from output_layer import *
import math
from activationFunctions import *

HIDDEN_LAYER = 2*int(math.sqrt(28*28))

class NeuralNetwork:
	def __init__(self, input_values):
		self.nn_inputs = input_values
		self.output = ""
		self.maxIndex = 0
		self.outlay = ''
		self.inlay = ''
		self.hidlay = ''
		self.output_layer_output = []
		self.hidden_layer_outputs = []
		self.input_layer_outputs = []
		self.expected = []
		self.work()

	def work(self):
		choice = input("User GD or SGD? (1/2)");
		if choice=='1':
			self.calcExpected()
			self.createNetwork()
			self.trainNetwork()
			# self.calcOutput(self.maxIndex)
		elif choice=='2':
			self.sgdCreateNetworkAndTrain()

	def createNetwork(self):
		print("Creating the newtork")
		inlay_inputs = [x[1] for x in self.nn_inputs]
		# print(len(inlay_inputs))
		self.inlay = InputLayer(inlay_inputs)
		self.input_layer_outputs = self.inlay.getOutput()
		self.hidlay = HiddenLayer(HIDDEN_LAYER, self.input_layer_outputs)
		self.hidden_layer_outputs = self.hidlay.getOutput()
		self.outlay = OutputLayer(10, self.hidden_layer_outputs)
		self.output_layer_output = self.outlay.getOutput()
		print(self.output_layer_output)
		print(self.output_layer_output[0])

	def calcExpected(self):
		print("Calculation Expected Outputs")
		for value in self.nn_inputs:
			self.expected.append(value[0])

		# print(self.expected)

	def calcError(self):
		real = []
		e = 0

		for value in self.output_layer_output:
			ind = calcMaxIndex(value)
			out = calcOutput(value)
			real.append(out)

		error = ord(self.expected) - ord(real)
		for value in error:
			e += int(value)

		return (error, e)

	def calcMaxIndex(self, li):
		return li.index(max(li))


	def calcOutput(self, x):
		x = int(x)
		self.output = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J'}[x]
		return self.output


	def trainNetwork(self):
		num = 100
		while(num > 20):
			l2_error , num = self.calcError()
			# l2_delta = l2_error*sigmoidDeriv()



	def getOutput(self):
		return self.output


	def sgdTrain(self, expected, value):
		print("In training")
		error = self.sgdCalcError(expected, value)
		r = 0
		while(r<4):
			k = 0
			r +=1 
			if(r==1):
				c = 0
			elif(r==2):
				c = 1
			elif(r==3):
				c = 2
			imageInd = 0
			while(imageInd < len(self.nn_inputs)):
				imageInd = 3*k + c
				self.input_layer_outputs = self.inlay.setNewInput(self.nn_inputs[imageInd][1])
				self.hidden_layer_outputs = self.hidlay.setNewInput(self.input_layer_outputs)
				self.output_layer_output = self.outlay.setNewInput(self.hidden_layer_outputs)
				print("Expected : {}".format(self.nn_inputs[imageInd][0]))
				print("Got : "+self.calcOutput(self.calcMaxIndex(self.output_layer_output)))

				l2_error = self.sgdCalcError(self.nn_inputs[imageInd][0], self.calcOutput(self.calcMaxIndex(self.output_layer_output)))
				l2_delta = l2_error*sigmoidDeriv(self.output_layer_output[self.calcMaxIndex(self.output_layer_output)])
				syn1 = self.out.getWeights()
				l1_error = l2_delta.dot(syn1.T)
				print(syn1)
				print(l2_error)
				print(l2_delta)

				k +=1



	def sgdCalcError(self, expected, value):
		# print(type(expected))
		expected = ord(expected)
		# print(value)
		# print(type(value))
		value = ord(value)
		return abs(expected - value)

	def sgdCreateNetworkAndTrain(self):
		inlay_inputs = self.nn_inputs[0][1]
		self.inlay = InputLayer(inlay_inputs, 2)
		self.input_layer_outputs = self.inlay.getOutput()
		self.hidlay = HiddenLayer(HIDDEN_LAYER, self.input_layer_outputs)
		self.hidden_layer_outputs = self.hidlay.getOutput()
		self.outlay = OutputLayer(10, self.hidden_layer_outputs)
		self.output_layer_output = self.outlay.getOutput()
		self.sgdTrain(self.nn_inputs[0][0], self.calcOutput(self.calcMaxIndex(self.output_layer_output)))

		






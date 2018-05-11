from input_layer import *
from hidden_layer import *
from output_layer import *
import math

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
		self.calcExpected()
		self.createNetwork()
		self.trainNetwork()
		# self.calcOutput(self.maxIndex)

	def createNetwork(self):
		print("Creating the newtork")
		inlay_inputs = [x[1] for x in self.nn_inputs]
		# print(len(inlay_inputs))
		self.inlay = InputLayer(inlay_inputs)
		self.input_layer_outputs = self.inlay.getOutput()
		self.hidlay = HiddenLayer(HIDDEN_LAYER, input_layer_outputs)
		self.hidden_layer_outputs = self.hidlay.getOutput()
		self.outlay = OutputLayer(10, hidden_layer_outputs)
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

		error = self.expected - real
		for value in error:
			e += int(value)

		return (error, e)

	def calcMaxIndex(self, li):
		return li.index(max(li))


	def calcOutput(self, x):
		self.output = {'0':'A', '1':'B',  '2':'C', '3':'D', '4':'E', '5':'F', '6':'G', '7':'H', '8':'I', '9':'J'}[x]


	def trainNetwork(self):
		num = 100
		while(num > 20):
			l2_error , num = self.calcError()
			# l2_delta = l2_error*sigmoidDeriv()



	def getOutput(self):
		return self.output





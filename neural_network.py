from input_layer import *
from hidden_layer import *
from output_layer import *
import math

class NeuralNetwork:
	def __init__(self, input_values):
		self.nn_inputs = input_values
		self.output = ""
		self.maxIndex = 0
		self.outlay = ''
		self.inlay = ''
		self.hidlay = ''
		self.work()

	def work(self):
		self.createNetwork()
		self.trainNetwork()
		self.calcOutput(self.maxIndex)

	def createNetwork(self):
		self.inlay = InputLayer(self.nn_inputs)
		input_layer_outputs = self.inlay.getOutput()
		self.hidlay = HiddenLayer(int(math.sqrt(len(input_layer_outputs))), input_layer_outputs)
		hidden_layer_outputs = self.hidlay.getOutput()
		self.outlay = OutputLayer(10, hidden_layer_outputs)
		output_layer_output = self.outlay.getOutput()
		self.maxIndex = self.calcMaxIndex(output_layer_output)


	def moreThanOneAboveHalf(self, li):
		ans = [x for x in li if x > 0.5]
		if len(ans) > 1:
			return True
		return False


	def calcMaxIndex(self, li):
		return li.index(max(li))


	def calcOutput(self, x):
		self.output = {'0':'A', '1':'B',  '2':'C', '3':'D', '4':'E', '5':'F', '6':'G', '7':'H', '8':'I', '9':'J'}[x]


	def trainNetwork(self):
		while(self.moreThanOneAboveHalf(self.outlay.getOutput())):
			pass

		self.calcOutput()


	def getOutput(self):
		return self.output





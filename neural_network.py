from input_layer import *
from hidden_layer import *
from output_layer import *
import math
from activationFunctions import *

HIDDEN_LAYER = 2*int(math.sqrt(28*28))
ROUND = 4

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
			#self.calcExpected()
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
		#print(self.output_layer_output)
		#print(self.output_layer_output[0])

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

		error =  (self.expected) - ord(real)
		for value in error:
			e += int(value)

		return (error, e)

	def calcMaxIndex(self, li):
		return li.index(max(li))


	def calcOutput(self, x):
		#print("Here")
		#print(x)
		x = int(x)
		#print("x -> ",x)
		self.output = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J'}[x]
		return self.output


	def trainNetwork(self):
		num = 100
		while(num > 20):
			l2_error , num = self.calcError()
			# l2_delta = l2_error*sigmoidDeriv()



	def getOutput(self):
		return self.output

	# def getOutForInput(self , x):
	# 	if self.input_layer_outputs[x] == 'A':
	# 		return 


	def sgdTrain(self):
		print("In training")
		#print (expected , value)
		#error = self.sgdCalcError(expected, value)
		turn  = 0
		while(turn<ROUND):
			#print("I'm now in while(turn<ROUND) ")
			coef = 0
			turn +=1 
			if(turn==1):
				remainder = 0
			elif(turn==2):
				remainder = 1
			elif(turn==3):
				remainder = 2
			imageInd = 0
			#print("len of nn_inputs: " , len(self.nn_inputs))
			while(imageInd < len(self.nn_inputs)):
				#print("And Now in Second while:)")
				#print("this is imageInd: " , imageInd)
				#print("imageInd: ",self.nn_inputs[imageInd][1])
				#print("sgd size input : "  , len(self.inlay.setNewInput(self.nn_inputs[imageInd][1])))
				self.input_layer_outputs = self.inlay.setNewInput(self.nn_inputs[imageInd][1])
				#print("sgd size input out : "  , len(self.input_layer_outputs))
				#print ("this is the len of input_layer_outputs -> ",len(self.input_layer_outputs))
				#print ("out input layer: ",self.input_layer_outputs)
				#print("hello sabri:)")
				self.hidden_layer_outputs = self.hidlay.setNewInput(self.input_layer_outputs)
				#print ("out hidden layer: " , self.hidden_layer_outputs)
				self.output_layer_output = self.outlay.setNewInput(self.hidden_layer_outputs)
				#print("Expected : {}".format(self.nn_inputs[imageInd][0]))
				#print("Got : "+self.calcOutput(self.calcMaxIndex(self.output_layer_output)))
				l2_error = self.sgdCalcError(self.nn_inputs[imageInd][0], self.calcOutput(self.calcMaxIndex(self.output_layer_output)))
				#print("output layer output -> "  , self.calcOutput(self.calcMaxIndex(self.output_layer_output)))
				l2_delta = l2_error*sigmoidDeriv(ord(self.calcOutput(self.calcMaxIndex(self.output_layer_output))))
				#print("max index out layer: "  , self.calcMaxIndex(self.output_layer_output))
				#print("sgdtrain out layer: " , self.calcOutput(self.calcMaxIndex(self.output_layer_output)))
				#TODO: Test
				syn2 = self.outlay.getWeights()
				syn1 = self.hidlay.getWeights()
				syn0 = self.inlay.getWeights()
				#print(syn0)
				# print("syn1.T is: " , syn1.T)
				l2 = ord(self.calcOutput(self.calcMaxIndex(self.output_layer_output)))
				l1 = ord(self.calcOutput(self.calcMaxIndex(self.hidden_layer_outputs)))
				#l0 = ord(self.calcOutput(self.calcMaxIndex(self.input_layer_outputs)))
				#print("sgdtrain hidden layer" , self.calcOutput(self.calcMaxIndex(self.hidden_layer_outputs)))
				#print("max index input layer: " , self.calcMaxIndex(self.input_layer_outputs))
				#m = self.calcMaxIndex(self.input_layer_outputs)
				#print("sgdtrain input layer"  , self.input_layer_outputs[self.calcMaxIndex(self.input_layer_outputs)])
				#print("calc output input layer: " , self.calcOutput(m))
				l1_error = np.asarray(l2_delta).dot(syn2.T)
				l1_delta = l1_error*sigmoidDeriv(l1)
				# print(l1.T)
				syn2 += np.asarray(l1).T.dot(np.asarray(l2_delta))
				#print("syn2: "  , syn2)
				#syn1 += l0.T.dot(np.asarray(l1_delta))

				# print(syn1)
				# print(syn2)
				# print(l2_error)
				# print(l2_delta)
				# print(l1_error)
				# print(l1_delta)

				coef +=1
				imageInd = 3*coef + remainder



	def sgdCalcError(self, expected, value):
		# print(type(expected))
		expected = ord(expected)
		# print(value)
		# print(type(value))
		value = ord(value)
		return abs(expected - value)

	def sgdCreateNetworkAndTrain(self):
		inlay_inputs = self.nn_inputs[0][1]
		#print("this is inlay input in sgdCreateNetworkAndTrain",inlay_inputs)
		self.inlay = InputLayer(inlay_inputs, 2)
		#print("self inlay in sgdCreateNetworkAndTrain" , self.inlay)
		#print("size inlay_inputs: ",len(self.inlay.getOutput()))
		self.input_layer_outputs = self.inlay.getOutput()
		#print("size input_layer_outputs: ",len(self.input_layer_outputs))
		#print("input layer out  in sgdCreateNetworkAndTrain: " , self.input_layer_outputs)
		self.hidlay = HiddenLayer(HIDDEN_LAYER, self.input_layer_outputs)
		#print("hidden layer out  in sgdCreateNetworkAndTrain: " , self.hidlay)
		self.hidden_layer_outputs = self.hidlay.getOutput()
		#print("HIDDEN layer out  in sgdCreateNetworkAndTrain: " , self.hidden_layer_outputs)
		self.outlay = OutputLayer(10, self.hidden_layer_outputs)
		self.output_layer_output = self.outlay.getOutput()
		#print("output layer out  in sgdCreateNetworkAndTrain: " , self.output_layer_output)
		#print("max index in out of output layer: " , self.calcMaxIndex(self.output_layer_output))
		#print("3 index in out layer: " , self.output_layer_output[3])
		#print("self.nn_inputs[0][0]: " , self.nn_inputs[0][0])
		#print("out of calcOutput: ",self.calcOutput(self.calcMaxIndex(self.output_layer_output)))
		self.sgdTrain()

		






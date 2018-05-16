from input_layer import *
from hidden_layer import *
from output_layer import *
import math
from activationFunctions import *
import random 

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
		self.test = []
		self.choice = 0
		self.work()

	def work(self):
		self.separateTestData()
		self.choice = input("User GD or SGD? (1/2)");
		print("I am your choice: "  , self.choice)
		if self.choice=='1':
			self.createNetworkGD()
			self.trainNetworkGD()
		elif self.choice=='2':
			self.sgdCreateNetworkAndTrain()

	def separateTestData(self):
		for i in range(2000):
			ind = random.randint(0, len(self.nn_inputs))
			self.test.append(self.nn_inputs[ind])
			del self.nn_inputs[ind]

	def createNetworkGD(self):
		print("Creating the newtork")
		inlay_inputs = self.nn_inputs[0][1]
		self.inlay = InputLayer(inlay_inputs, 2)
		self.input_layer_outputs = self.inlay.getOutput()
		self.hidlay = HiddenLayer(HIDDEN_LAYER, self.input_layer_outputs)
		self.hidden_layer_outputs = self.hidlay.getOutput()
		self.outlay = OutputLayer(10, self.hidden_layer_outputs)
		self.output_layer_output = self.outlay.getOutput()
		self.trainNetworkGD()

	# def calcExpected(self):
	# 	print("Calculation Expected Outputs")
	# 	for value in self.nn_inputs:
	# 		self.expected.append(value[0])


	# def calcError(self):
	# 	real = []
	# 	e = 0

	# 	for value in self.output_layer_output:
	# 		ind = calcMaxIndex(value)
	# 		out = calcOutput(value)
	# 		real.append(out)

	# 	error =  (self.expected) - ord(real)
	# 	for value in error:
	# 		e += int(value)

	# 	return (error, e)

	def calcMaxIndex(self, li):
		return li.index(max(li))


	def calcOutput(self, x):
		x = int(x)
		self.output = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J'}[x]
		return self.output


	def trainNetworkGD(self):

		totalError = 0

		for img in self.nn_inputs:
			self.input_layer_outputs = self.inlay.setNewInput(img[1])
			self.hidden_layer_outputs = self.hidlay.setNewInput(self.input_layer_outputs)
			self.output_layer_output = self.outlay.setNewInput(self.hidden_layer_outputs)

			totalError += self.sgdCalcError(img[0], self.calcOutput(self.calcMaxIndex(self.output_layer_output)))

		self.updateWeights(totalError)


	def runTests(self):
		wrongAnswers = 0

		for img in self.test:
			self.input_layer_outputs = self.inlay.setNewInput(img[1])
			self.hidden_layer_outputs = self.hidlay.setNewInput(self.input_layer_outputs)
			self.output_layer_output = self.outlay.setNewInput(self.hidden_layer_outputs)

			error = self.sgdCalcError(img[0], self.calcOutput(self.calcMaxIndex(self.output_layer_output)))
			if not error ==0:
				wrongAnswers +=1 


		errorPercent = (wrongAnswers/len(self.test))*100
		print("Error Percentage: {}".format(errorPercent))
		if int(errorPercent) > 10:
			print ("in the if:D")
			if self.choice=='1':
				self.trainNetworkGD()
			elif self.choice=='2':
				self.sgdTrain()




	def getOutput(self):
		return self.output


	def sgdTrain(self):
		print("In training")
		turn  = 0
		while(turn<ROUND):
			coef = 0
			turn +=1 
			if(turn==1):
				remainder = 0
			elif(turn==2):
				remainder = 1
			elif(turn==3):
				remainder = 2
			imageInd = remainder
			while(imageInd < len(self.nn_inputs)):
				if random.randint(0, 100) == 1:
					delta = random.randint(0, HIDDEN_LAYER)
					self.hidlay.dropOut(delta)
				self.input_layer_outputs = self.inlay.setNewInput(self.nn_inputs[imageInd][1])
				self.hidden_layer_outputs = self.hidlay.setNewInput(self.input_layer_outputs)
				self.output_layer_output = self.outlay.setNewInput(self.hidden_layer_outputs)

				# print("Expected : {}".format(self.nn_inputs[imageInd][0]))
				# print("Got : "+self.calcOutput(self.calcMaxIndex(self.output_layer_output)))

				l2_error = self.sgdCalcError(self.nn_inputs[imageInd][0], self.calcOutput(self.calcMaxIndex(self.output_layer_output)))
				if not l2_error==0:
					self.updateWeights(l2_error)
				
				coef +=1
				imageInd = 3*coef + remainder
		print("Finish training!")
		self.runTests()





	def updateWeights(self, l2_error):

		#print("output layer output -> "  , self.calcOutput(self.calcMaxIndex(self.output_layer_output)))
		l2 = ord(self.calcOutput(self.calcMaxIndex(self.output_layer_output)))
		l2_delta = l2_error*sigmoidDeriv(l2)
		#print("max index out layer: "  , self.calcMaxIndex(self.output_layer_output))
		#print("sgdtrain out layer: " , self.calcOutput(self.calcMaxIndex(self.output_layer_output)))
		#TODO: Test
		syn2 = self.outlay.getWeights()
		syn1 = self.hidlay.getWeights()
		#print(syn0)
		# print("syn1.T is: " , syn1.T)
		
		l1 = ord(self.calcOutput(self.calcMaxIndex(self.hidden_layer_outputs)))
		l0 = self.input_layer_outputs[self.calcMaxIndex(self.input_layer_outputs)]

		#print("sgdtrain hidden layer" , self.calcOutput(self.calcMaxIndex(self.hidden_layer_outputs)))
		#print("max index input layer: " , self.calcMaxIndex(self.input_layer_outputs))
		#m = self.calcMaxIndex(self.input_layer_outputs)
		#print("sgdtrain input layer"  , self.input_layer_outputs[self.calcMaxIndex(self.input_layer_outputs)])
		#print("calc output input layer: " , self.calcOutput(m))

		l1_error = np.asarray(l2_delta).dot(syn2.T)
		l1_delta = l1_error*sigmoidDeriv(l1)

		syn2 += np.asarray(l1).T.dot(np.asarray(l2_delta))
		print("len syn2: "  , syn2.size)
		# print ("size l1-delta: " , len(l1_delta))
		# syn1 += np.asarray(l0).T.dot(np.asarray(l1_delta))

		self.hidlay.setWeightsUpdated(syn1)
		self.outlay.setWeightsUpdated(syn2)
		# print(syn1)
		# print(syn2)
		# print(l2_error)
		# print(l2_delta)
		# print(l1_error)
		# print(l1_delta)



	def sgdCalcError(self, expected, value):
		expected = ord(expected)
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
		self.sgdTrain()

		






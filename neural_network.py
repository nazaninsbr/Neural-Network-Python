from input_layer import *
from hidden_layer import *
from output_layer import *
import math
from activationFunctions import *
import random 
import plot

HIDDEN_LAYER = 2*int(math.sqrt(28*28))
ROUND = 4
N = 0.5
M = 0.2

def makeMatrix(I, J, fill=0.0):
	m = []
	for i in range(I):
		m.append([fill]*J)
	return m

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
		self.ch = makeMatrix(HIDDEN_LAYER, 28*28)
		self.co = makeMatrix(10, HIDDEN_LAYER)
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
			ind = random.randint(0, len(self.nn_inputs))%len(self.nn_inputs)
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
		print("Training....")
		output_deltas = [0.0] * len(self.output_layer_output)
		totalError = 0

		for img in self.nn_inputs:
			self.input_layer_outputs = self.inlay.setNewInput(img[1])
			self.hidden_layer_outputs = self.hidlay.setNewInput(self.input_layer_outputs)
			self.output_layer_output = self.outlay.setNewInput(self.hidden_layer_outputs)

			expectedOutput = img[0]
			outputToIndex = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'J':9}[expectedOutput]
				
			for k in range(len(self.output_layer_output)):
				if not k ==outputToIndex:
					error = self.sgdCalcError('0', self.calcOutput(k))*self.output_layer_output[k]
				elif k ==outputToIndex:
					error = self.sgdCalcError(expectedOutput, self.calcOutput(k))*self.output_layer_output[k]
				output_deltas[k] += sigmoidDeriv(self.output_layer_output[k]) * error


			totalError += self.sgdCalcError(img[0], self.calcOutput(self.calcMaxIndex(self.output_layer_output)))

		print("Error: {}".format(totalError))
		print(output_deltas)
		self.updateWeights(output_deltas)
		self.runTests()


	def runTests(self):
		print("Running Tests: ")
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

		else:
			plot.mainFunc(self.hidlay.getWeights(), self.outlay.getWeights())





	def getOutput(self):
		return self.output


	def sgdTrain(self):
		print("In training")
		turn  = 0
		dropOutList = []
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
					print("dropping out!")
					delta = random.randint(0, HIDDEN_LAYER)%HIDDEN_LAYER
					dropOutList.append(delta)
					self.hidlay.dropOut(delta)

				self.input_layer_outputs = self.inlay.setNewInput(self.nn_inputs[imageInd][1])
				self.hidden_layer_outputs = self.hidlay.setNewInput(self.input_layer_outputs)
				self.output_layer_output = self.outlay.setNewInput(self.hidden_layer_outputs)

				# print("Expected : {}".format(self.nn_inputs[imageInd][0]))
				# print("Got : "+self.calcOutput(self.calcMaxIndex(self.output_layer_output)))
				# print(self.output_layer_output)
				ThisTimesError = 0.0
				expectedOutput = self.nn_inputs[imageInd][0]
				outputToIndex = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'J':9}
				output_deltas = [0.0] * len(self.output_layer_output)
				# print("output_layer_output: ", len(self.output_layer_output))
				# print("output_deltas: ", len(output_deltas))
				
				for k in range(len(self.output_layer_output)):
					if k > 9 :
						break
					if not k ==outputToIndex:
						error = self.sgdCalcError('0', self.calcOutput(k))*self.output_layer_output[k]
						ThisTimesError += 0.5*(error)**2
					elif k ==outputToIndex:
						error = self.sgdCalcError(expectedOutput, self.calcOutput(k))*self.output_layer_output[k]
					output_deltas[k] = sigmoidDeriv(self.output_layer_output[k]) * error
					if error>0 and output_deltas[k]==0:
						output_deltas[k] += 0.4
					# print("Error: ", error)

				
				# print(output_deltas)

				# print("Error: {}".format(ThisTimesError))

				self.updateWeights(output_deltas)
				
				coef +=1
				imageInd = 3*coef + remainder
				for i in dropOutList:
					self.hidlay.resetDropOut(i)
				del dropOutList[:]

		print("Finish training!")
		self.runTests()



	def updateWeights(self, output_deltas):

		# calculate error terms for hidden
		hidden_deltas = [0.0] * HIDDEN_LAYER
		wo = self.outlay.getWeights()
		wh = self.hidlay.getWeights()
		# print("weights before change: ", wo[0])
		# print("wo : ", len(wo))
		# print("wh : ", len(wh))
		# print("output_deltas: ", len(output_deltas))
		for j in range(HIDDEN_LAYER):
			error = 0.0
			for k in range(len(self.output_layer_output)):
				# print("index of output_layer_output: ", k)
				# print("index of hidden_layer_outputs: ", j)
				error = error + output_deltas[k]*wo[k][j]
			hidden_deltas[j] = sigmoidDeriv(self.hidden_layer_outputs[j]) * error
			# print("Error: ", error)
			# print("hidden_deltas[j]: ", hidden_deltas[j])

		# update output weights
		for j in range(len(self.output_layer_output)):
			for k in range(len(self.hidden_layer_outputs)):
				# print("index of output_layer_output: ", j)
				# print("index of hidden_layer_outputs: ", k)
				# print("calculating change, output_deltas: ", output_deltas[j])
				# print("calculating change, hidden_layer_outputs: ",self.hidden_layer_outputs[k])
				change = output_deltas[j]*self.hidden_layer_outputs[k]
				# print("j , k before change: ", wo[j][k])
				# wo[j][k] = wo[j][k] + N*change
				# print("change val: ", N*change)
				# print("j , k after change: ", wo[j][k])
				wo[j][k] = wo[j][k] + N*change + M*self.co[j][k]
				self.co[j][k] = change
				#print N*change, M*self.co[j][k]

		# update hidden weights
		for i in range(HIDDEN_LAYER):
			for j in range(len(self.input_layer_outputs)):
				change = hidden_deltas[i]*self.input_layer_outputs[j]
				# wh[i][j] = wh[i][j] + N*change
				wh[i][j] = wh[i][j] + N*change + M*self.ch[i][j]
				self.ch[i][j] = change

		# print("weights after change: ", wo[0])

		self.outlay.setWeightsUpdated(wo)
		self.hidlay.setWeightsUpdated(wh)



	def sgdCalcError(self, expected, value):
		# if not type(expected)==int:
		expected = ord(expected)
		value = ord(value)
		return expected - value

	def sgdCreateNetworkAndTrain(self):
		inlay_inputs = self.nn_inputs[0][1]
		self.inlay = InputLayer(inlay_inputs, 2)
		self.input_layer_outputs = self.inlay.getOutput()
		self.hidlay = HiddenLayer(HIDDEN_LAYER, self.input_layer_outputs)
		self.hidden_layer_outputs = self.hidlay.getOutput()
		self.outlay = OutputLayer(10, self.hidden_layer_outputs)
		self.output_layer_output = self.outlay.getOutput()
		self.sgdTrain()


		






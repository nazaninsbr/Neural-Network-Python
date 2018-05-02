import math 

LINEAR_A = 2
LINEAR_B = 3

def linear1(inputValue):
	return inputValue

def linearAxB(inputValue):
	return LINEAR_A*inputValue + LINEAR_B

def sigmoid(inputValue):
	return 1/(1+ math.exp(-1*inputValue))

def stepFunction(inputValue):
	if inputValue < 0:
		return 0
	return 1

def reloFunction(inputValue):
	if inputValue < 0:
		return 0
	return inputValue
import math 

LINEAR_A = 2
LINEAR_B = 3

def linear1(inputValue):
	return inputValue

def linearAxB(inputValue):
	return LINEAR_A*inputValue + LINEAR_B

def sigmoid(inputValue):
	try:
		ans = 1/(1+ math.exp(-1*inputValue))
	except OverflowError:
		ans = 1/math.exp(-1*(inputValue%10))
	return ans

def stepFunction(inputValue):
	if inputValue < 0:
		return 0
	return 1

def reloFunction(inputValue):
	if inputValue < 0:
		return 0
	return inputValue


def sigmoidDeriv(inputValue):
	return inputValue*(1-inputValue)


def linearDeriv(inputValue):
	return LINEAR_A
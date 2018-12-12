import numpy as np

def sigmoid (x):
	return 1/(1 + np.exp(-x))

def derivatives_sigmoid(x):
	val = sigmoid(x)
	return val * (1-val)

def linear(x):
	return x

def derivatives_linear(x):
	return 1

def tanh(x):
	return np.tanh(x)

def derivatives_tanh(x):
	return 1 - np.tanh(x)**2

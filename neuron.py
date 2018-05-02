from activationFunctions import *

class Neuron:
	def __init__(self, pixel_value, weight, bias):
		self.pixel_value = pixel_value
		self.weight = weight
		self.bias = bias
		self.sigma = 0
		self.output = 0

	def calc_sigma(self):
		if str(type(self.pixel_value))=="<class 'numpy.float32'>":
			self.sigma += self.weight*self.pixel_value
		else:
			for pixel in self.pixel_value:
				self.sigma += self.weight*pixel


	def calc_activation(self):
		self.output = linear1(self.sigma)


	def calc_output(self):
		self.calc_sigma()
		self.calc_activation()
		return self.output
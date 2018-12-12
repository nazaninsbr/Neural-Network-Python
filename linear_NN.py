from activation_functions import *
import numpy as np
import matplotlib.pyplot as plt

USE_MOMENTUM = True

def get_next_batch(dataset, result, last_input_index, batch_size):
	start_index = (last_input_index+1)%len(dataset)
	this_epochs_inputs = np.array(dataset[start_index: start_index+batch_size])
	this_epochs_labels = np.array(result[start_index: start_index+batch_size])
	return last_input_index+batch_size, this_epochs_inputs, this_epochs_labels


def gradient_descent(dataset, result, out_w, out_b, iteration_length, r, lr, batch_size):
	y_axis = []
	x_axis = []
	last_input_index = -1
	
	for k in range (iteration_length):
		last_input_index, this_epochs_dataset, this_epochs_results = get_next_batch(dataset, result, last_input_index, batch_size)

		out_amount = this_epochs_dataset.dot(out_w) + out_b
		out_amount_active = sigmoid(out_amount)

		E = this_epochs_results-out_amount_active
		slope_output_layer = derivatives_sigmoid(out_amount_active)
		d_output = E * slope_output_layer

		if USE_MOMENTUM==True:
			out_w += (this_epochs_dataset.T.dot(d_output) + out_w * r ) * lr
		elif USE_MOMENTUM==False:
			out_w += (this_epochs_dataset.T.dot(d_output)) * lr
		out_b += d_output.sum(axis=0) * lr
		
		cost = np.mean(np.square(this_epochs_results - out_amount_active))
		print ("[",k,"]: ",cost)

		y_axis.append(cost)
		x_axis.append(k)
	plt.plot(x_axis,y_axis)
	plt.show()    
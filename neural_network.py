from activation_functions import *
import numpy as np
import matplotlib.pyplot as plt

USE_MOMENTUM = True
# 1 sigmoid 2 tanh 3 linear
ACTIVATION_FUNCTION_CHOICE = 1

def get_next_batch(dataset, result, last_input_index, batch_size):
	start_index = (last_input_index+1)%len(dataset)
	this_epochs_inputs = np.array(dataset[start_index: start_index+batch_size])
	this_epochs_labels = np.array(result[start_index: start_index+batch_size])
	return last_input_index+batch_size, this_epochs_inputs, this_epochs_labels


def gradient_descent(dataset, result, hid_w, out_w, hid_b, out_b, iteration_length, r, lr, batch_size):
	y_axis = []
	x_axis = []
	last_input_index = -1
	
	for k in range (iteration_length):
		last_input_index, this_epochs_dataset, this_epochs_results = get_next_batch(dataset, result, last_input_index, batch_size)

		hid_amount = this_epochs_dataset.dot(hid_w) + hid_b
		if ACTIVATION_FUNCTION_CHOICE==1:
			hid_amount_active = sigmoid(hid_amount)
		elif ACTIVATION_FUNCTION_CHOICE==2:
			hid_amount_active = tanh(hid_amount)
		elif ACTIVATION_FUNCTION_CHOICE==3:
			hid_amount_active = linear(hid_amount)
		out_amount = hid_amount_active.dot(out_w) + out_b
		if ACTIVATION_FUNCTION_CHOICE==1:
			out_amount_active = sigmoid(out_amount)
		elif ACTIVATION_FUNCTION_CHOICE==2:
			out_amount_active = tanh(out_amount)
		elif ACTIVATION_FUNCTION_CHOICE==3:
			out_amount_active = linear(out_amount)

		E = this_epochs_results-out_amount_active
		if ACTIVATION_FUNCTION_CHOICE==1:
			slope_output_layer = derivatives_sigmoid(out_amount_active)
			slope_hidden_layer = derivatives_sigmoid(hid_amount_active)
		elif ACTIVATION_FUNCTION_CHOICE==2:
			slope_output_layer = derivatives_tanh(out_amount_active)
			slope_hidden_layer = derivatives_tanh(hid_amount_active)
		elif ACTIVATION_FUNCTION_CHOICE==3:
			slope_output_layer = derivatives_linear(out_amount_active)
			slope_hidden_layer = derivatives_linear(hid_amount_active)

		d_output = E * slope_output_layer
		Error_at_hidden_layer = d_output.dot(out_w.T)
		d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer

		if USE_MOMENTUM==True:
			out_w += (hid_amount_active.T.dot(d_output) + out_w * r ) * lr
		elif USE_MOMENTUM==False:
			out_w += (hid_amount_active.T.dot(d_output)) * lr
		out_b += d_output.sum(axis=0) * lr
		if USE_MOMENTUM==True:
			hid_w += (this_epochs_dataset.T.dot(d_hiddenlayer) + hid_w * r) * lr
		elif USE_MOMENTUM==False:
			hid_w += (this_epochs_dataset.T.dot(d_hiddenlayer)) * lr
		hid_b += d_hiddenlayer.sum(axis = 0) * lr

		cost = np.mean(np.square(this_epochs_results - out_amount_active))
		print ("[",k,"]: ",cost)

		y_axis.append(cost)
		x_axis.append(k)
	plt.plot(x_axis,y_axis)
	plt.show()    


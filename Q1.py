from data_readers import *
from activation_functions import *
from normalization_functions import *
from neural_network import gradient_descent
from random import seed
from random import random
import numpy as np
import random
import matplotlib.pyplot as plt

TEST_IMAGES_DATASET = './Dataset/test_images.mat'
TRAIN_IMAGES_DATASET = './Dataset/train_images.mat'
TEST_LABELS_DATASET = './Dataset/test_labels.mat'
TRAIN_LABELS_DATASET = './Dataset/train_lables.mat'
in_layer_size = 784
out_layer_size_binary_format = TOTAL_NUMBER_OF_BINARY_BITS = 4
out_layer_size = 10

def read_the_data():
	train_images = read_mat_file(TRAIN_IMAGES_DATASET, "train_images")
	train_labels = read_mat_file(TRAIN_LABELS_DATASET, "train_lables")
	test_images = read_mat_file(TEST_IMAGES_DATASET, "test_images")
	test_labels = read_mat_file(TEST_LABELS_DATASET, "test_labels")
	return train_images, train_labels, test_images,test_labels

def convert_labels_to_one_hot(labels):
	retLabels = []
	for result in labels:
		thisImagesLabel = []
		for i in range(0, 10):
			if result==i:
				thisImagesLabel.append(1)
			else:
				thisImagesLabel.append(0)
		retLabels.append(thisImagesLabel)
	return retLabels

def convert_labels_to_binary(labels, TOTAL_NUMBER_OF_BINARY_BITS):
	retLabels = []
	for result in labels:
		format_str = '{0:0' +str(TOTAL_NUMBER_OF_BINARY_BITS) + 'b}'
		binary = format_str.format(result)
		retLabels.append([int(x) for x in binary])
	return retLabels

def train_data_accuracy(train_images, train_labels, hid_w, hid_b, out_w, out_b):
	dataset= np.array(train_images)
	result = np.array(convert_labels_to_one_hot(train_labels[0]))
	hid_amount = dataset.dot(hid_w) + hid_b
	hid_amount_active = sigmoid(hid_amount)
	out_amount = hid_amount_active.dot(out_w) + out_b
	out_amount_active = (out_amount)
	cnt = 0
	for i in  range (len(result)):
		if (np.argmax(result[i]) == np.argmax(out_amount_active[i])):
			cnt += 1
	print("Training Acc= ", cnt/len(result) * 100, "%")

def train_data_accuracy_with_binary_format(train_images, train_labels, hid_w, hid_b, out_w, out_b):
	dataset= np.array(train_images)
	result = np.array(convert_labels_to_binary(train_labels[0]))
	hid_amount = dataset.dot(hid_w) + hid_b
	hid_amount_active = sigmoid(hid_amount)
	out_amount = hid_amount_active.dot(out_w) + out_b
	out_amount_active = (out_amount)
	cnt = 0
	for i in  range (len(result)):
		if (np.argmax(result[i]) == np.argmax(out_amount_active[i])):
			cnt += 1
	print("Training Acc= ", cnt/len(result) * 100, "%")

def test_data_accuracy_with_binary_format(test_images, test_labels, hid_w, hid_b, out_w, out_b):
	dataset2= np.array(test_images)
	result2 = np.array(convert_labels_to_binary(test_labels[0]))

	hid_amount = dataset2.dot(hid_w) + hid_b
	hid_amount_active = sigmoid(hid_amount)
	out_amount = hid_amount_active.dot(out_w) + out_b
	out_amount_active = (out_amount)
	cnt = 0
	for i in  range (len(result2)):
		if (np.argmax(result2[i]) == np.argmax(out_amount_active[i])):
			cnt += 1
	print("New Data Acc= ", cnt/len(result2) * 100, "%")

def test_data_accuracy(test_images, test_labels, hid_w, hid_b, out_w, out_b):
	dataset2= np.array(test_images)
	result2 = np.array(convert_labels_to_one_hot(test_labels[0]))

	hid_amount = dataset2.dot(hid_w) + hid_b
	hid_amount_active = sigmoid(hid_amount)
	out_amount = hid_amount_active.dot(out_w) + out_b
	out_amount_active = (out_amount)
	cnt = 0
	for i in  range (len(result2)):
		if (np.argmax(result2[i]) == np.argmax(out_amount_active[i])):
			cnt += 1
	print("New Data Acc= ", cnt/len(result2) * 100, "%")

def test_the_effect_of_different_epochs(train_images, train_labels, test_images, test_labels):
	hid_layer_size = 50
	r = 0.1
	batch_size = 60000
	lr = 0.00001

	hid_w = np.random.uniform(-0.1,0.1,(in_layer_size, hid_layer_size))
	out_w = np.random.uniform(-0.1,0.1,(hid_layer_size, out_layer_size))

	hid_b = np.ones(hid_layer_size)
	out_b = np.ones(out_layer_size)

	iteration_length = 500
	gradient_descent(train_images, convert_labels_to_one_hot(train_labels[0]), hid_w, out_w, hid_b, out_b, iteration_length, r, lr, batch_size)
	train_data_accuracy(train_images, train_labels, hid_w, hid_b, out_w, out_b)
	test_data_accuracy(test_images, test_labels, hid_w, hid_b, out_w, out_b)

	hid_w2 = np.random.uniform(-0.1,0.1,(in_layer_size, hid_layer_size))
	out_w2 = np.random.uniform(-0.1,0.1,(hid_layer_size, out_layer_size))

	hid_b2 = np.ones(hid_layer_size)
	out_b2 = np.ones(out_layer_size)

	iteration_length = 1000
	gradient_descent(train_images, convert_labels_to_one_hot(train_labels[0]), hid_w2, out_w2, hid_b2, out_b2, iteration_length, r, lr, batch_size)
	train_data_accuracy(train_images, train_labels, hid_w2, hid_b2, out_w2, out_b2)
	test_data_accuracy(test_images, test_labels, hid_w2, hid_b2, out_w2, out_b2)

	hid_w3 = np.random.uniform(-0.1,0.1,(in_layer_size, hid_layer_size))
	out_w3 = np.random.uniform(-0.1,0.1,(hid_layer_size, out_layer_size))

	hid_b3 = np.ones(hid_layer_size)
	out_b3 = np.ones(out_layer_size)

	iteration_length = 5000
	gradient_descent(train_images, convert_labels_to_one_hot(train_labels[0]), hid_w3, out_w3, hid_b3, out_b3, iteration_length, r, lr, batch_size)
	train_data_accuracy(train_images, train_labels, hid_w3, hid_b3, out_w3, out_b3)
	test_data_accuracy(test_images, test_labels, hid_w3, hid_b3, out_w3, out_b3)

def test_the_effect_of_different_learning_rates(train_images, train_labels, test_images,test_labels):
	hid_layer_size = 50
	r = 0.1
	batch_size = 60000
	iteration_length = 500

	hid_w = np.random.uniform(-0.1,0.1,(in_layer_size, hid_layer_size))
	out_w = np.random.uniform(-0.1,0.1,(hid_layer_size, out_layer_size))

	hid_b = np.ones(hid_layer_size)
	out_b = np.ones(out_layer_size)

	lr = 0.1
	gradient_descent(train_images, convert_labels_to_one_hot(train_labels[0]), hid_w, out_w, hid_b, out_b, iteration_length, r, lr, batch_size)
	train_data_accuracy(train_images, train_labels, hid_w, hid_b, out_w, out_b)
	test_data_accuracy(test_images, test_labels, hid_w, hid_b, out_w, out_b)

	hid_w2 = np.random.uniform(-0.1,0.1,(in_layer_size, hid_layer_size))
	out_w2 = np.random.uniform(-0.1,0.1,(hid_layer_size, out_layer_size))

	hid_b2 = np.ones(hid_layer_size)
	out_b2 = np.ones(out_layer_size)

	lr = 1
	gradient_descent(train_images, convert_labels_to_one_hot(train_labels[0]), hid_w2, out_w2, hid_b2, out_b2, iteration_length, r, lr, batch_size)
	train_data_accuracy(train_images, train_labels, hid_w2, hid_b2, out_w2, out_b2)
	test_data_accuracy(test_images, test_labels, hid_w2, hid_b2, out_w2, out_b2)

	hid_w3 = np.random.uniform(-0.1,0.1,(in_layer_size, hid_layer_size))
	out_w3 = np.random.uniform(-0.1,0.1,(hid_layer_size, out_layer_size))

	hid_b3 = np.ones(hid_layer_size)
	out_b3 = np.ones(out_layer_size)

	lr = 5
	gradient_descent(train_images, convert_labels_to_one_hot(train_labels[0]), hid_w3, out_w3, hid_b3, out_b3, iteration_length, r, lr, batch_size)
	train_data_accuracy(train_images, train_labels, hid_w3, hid_b3, out_w3, out_b3)
	test_data_accuracy(test_images, test_labels, hid_w3, hid_b3, out_w3, out_b3)

def test_the_effect_of_different_hidden_layer_size(train_images, train_labels, test_images,test_labels):
	r = 0.1
	lr = 0.00001
	batch_size = 60000
	iteration_length = 500

	hid_layer_size = 2
	hid_w = np.random.uniform(-0.1,0.1,(in_layer_size, hid_layer_size))
	out_w = np.random.uniform(-0.1,0.1,(hid_layer_size, out_layer_size))

	hid_b = np.ones(hid_layer_size)
	out_b = np.ones(out_layer_size)

	gradient_descent(train_images, convert_labels_to_one_hot(train_labels[0]), hid_w, out_w, hid_b, out_b, iteration_length, r, lr, batch_size)
	train_data_accuracy(train_images, train_labels, hid_w, hid_b, out_w, out_b)
	test_data_accuracy(test_images, test_labels, hid_w, hid_b, out_w, out_b)

	hid_layer_size = 150
	hid_w2 = np.random.uniform(-0.1,0.1,(in_layer_size, hid_layer_size))
	out_w2 = np.random.uniform(-0.1,0.1,(hid_layer_size, out_layer_size))

	hid_b2 = np.ones(hid_layer_size)
	out_b2 = np.ones(out_layer_size)

	gradient_descent(train_images, convert_labels_to_one_hot(train_labels[0]), hid_w2, out_w2, hid_b2, out_b2, iteration_length, r, lr, batch_size)
	train_data_accuracy(train_images, train_labels, hid_w2, hid_b2, out_w2, out_b2)
	test_data_accuracy(test_images, test_labels, hid_w2, hid_b2, out_w2, out_b2)

	hid_layer_size = 800
	hid_w3 = np.random.uniform(-0.1,0.1,(in_layer_size, hid_layer_size))
	out_w3 = np.random.uniform(-0.1,0.1,(hid_layer_size, out_layer_size))

	hid_b3 = np.ones(hid_layer_size)
	out_b3 = np.ones(out_layer_size)

	gradient_descent(train_images, convert_labels_to_one_hot(train_labels[0]), hid_w3, out_w3, hid_b3, out_b3, iteration_length, r, lr, batch_size)
	train_data_accuracy(train_images, train_labels, hid_w3, hid_b3, out_w3, out_b3)
	test_data_accuracy(test_images, test_labels, hid_w3, hid_b3, out_w3, out_b3)

def test_the_effect_of_different_momentum_values(train_images, train_labels, test_images,test_labels):
	r = 0.1
	lr = 0.00001
	batch_size = 60000
	iteration_length = 500
	hid_layer_size = 150

	hid_w = np.random.uniform(-0.1,0.1,(in_layer_size, hid_layer_size))
	out_w = np.random.uniform(-0.1,0.1,(hid_layer_size, out_layer_size))

	hid_b = np.ones(hid_layer_size)
	out_b = np.ones(out_layer_size)

	gradient_descent(train_images, convert_labels_to_one_hot(train_labels[0]), hid_w, out_w, hid_b, out_b, iteration_length, r, lr, batch_size)
	train_data_accuracy(train_images, train_labels, hid_w, hid_b, out_w, out_b)
	test_data_accuracy(test_images, test_labels, hid_w, hid_b, out_w, out_b)

	r = 8
	hid_w2 = np.random.uniform(-0.1,0.1,(in_layer_size, hid_layer_size))
	out_w2 = np.random.uniform(-0.1,0.1,(hid_layer_size, out_layer_size))

	hid_b2 = np.ones(hid_layer_size)
	out_b2 = np.ones(out_layer_size)

	gradient_descent(train_images, convert_labels_to_one_hot(train_labels[0]), hid_w2, out_w2, hid_b2, out_b2, iteration_length, r, lr, batch_size)
	train_data_accuracy(train_images, train_labels, hid_w2, hid_b2, out_w2, out_b2)
	test_data_accuracy(test_images, test_labels, hid_w2, hid_b2, out_w2, out_b2)

	r = 0.001
	hid_w3 = np.random.uniform(-0.1,0.1,(in_layer_size, hid_layer_size))
	out_w3 = np.random.uniform(-0.1,0.1,(hid_layer_size, out_layer_size))

	hid_b3 = np.ones(hid_layer_size)
	out_b3 = np.ones(out_layer_size)

	gradient_descent(train_images, convert_labels_to_one_hot(train_labels[0]), hid_w3, out_w3, hid_b3, out_b3, iteration_length, r, lr, batch_size)
	train_data_accuracy(train_images, train_labels, hid_w3, hid_b3, out_w3, out_b3)
	test_data_accuracy(test_images, test_labels, hid_w3, hid_b3, out_w3, out_b3)


def test_the_effect_of_different_initial_values(train_images, train_labels, test_images,test_labels):
	r = 0.001
	lr = 0.00001
	batch_size = 60000
	iteration_length = 500
	hid_layer_size = 150

	hid_w2 = np.random.uniform(-10,10,(in_layer_size, hid_layer_size))
	out_w2 = np.random.uniform(-10,10,(hid_layer_size, out_layer_size))

	hid_b2 = np.ones(hid_layer_size)
	out_b2 = np.ones(out_layer_size)

	gradient_descent(train_images, convert_labels_to_one_hot(train_labels[0]), hid_w2, out_w2, hid_b2, out_b2, iteration_length, r, lr, batch_size)
	train_data_accuracy(train_images, train_labels, hid_w2, hid_b2, out_w2, out_b2)
	test_data_accuracy(test_images, test_labels, hid_w2, hid_b2, out_w2, out_b2)

	hid_w3 = np.random.uniform(0,1,(in_layer_size, hid_layer_size))
	out_w3 = np.random.uniform(0,1,(hid_layer_size, out_layer_size))

	hid_b3 = np.ones(hid_layer_size)
	out_b3 = np.ones(out_layer_size)

	gradient_descent(train_images, convert_labels_to_one_hot(train_labels[0]), hid_w3, out_w3, hid_b3, out_b3, iteration_length, r, lr, batch_size)
	train_data_accuracy(train_images, train_labels, hid_w3, hid_b3, out_w3, out_b3)
	test_data_accuracy(test_images, test_labels, hid_w3, hid_b3, out_w3, out_b3)

def test_the_effect_of_different_batch_size(train_images, train_labels, test_images,test_labels):
	r = 0.001
	lr = 0.00001
	batch_size = 6
	iteration_length = 500
	hid_layer_size = 150

	hid_w = np.random.uniform(-0.1,0.1,(in_layer_size, hid_layer_size))
	out_w = np.random.uniform(-0.1,0.1,(hid_layer_size, out_layer_size))

	hid_b = np.ones(hid_layer_size)
	out_b = np.ones(out_layer_size)

	gradient_descent(train_images, convert_labels_to_one_hot(train_labels[0]), hid_w, out_w, hid_b, out_b, iteration_length, r, lr, batch_size)
	train_data_accuracy(train_images, train_labels, hid_w, hid_b, out_w, out_b)
	test_data_accuracy(test_images, test_labels, hid_w, hid_b, out_w, out_b)

	batch_size = 600
	hid_w2 = np.random.uniform(-0.1,0.1,(in_layer_size, hid_layer_size))
	out_w2 = np.random.uniform(-0.1,0.1,(hid_layer_size, out_layer_size))

	hid_b2 = np.ones(hid_layer_size)
	out_b2 = np.ones(out_layer_size)

	gradient_descent(train_images, convert_labels_to_one_hot(train_labels[0]), hid_w2, out_w2, hid_b2, out_b2, iteration_length, r, lr, batch_size)
	train_data_accuracy(train_images, train_labels, hid_w2, hid_b2, out_w2, out_b2)
	test_data_accuracy(test_images, test_labels, hid_w2, hid_b2, out_w2, out_b2)

	batch_size = 6000
	hid_w3 = np.random.uniform(-0.1,0.1,(in_layer_size, hid_layer_size))
	out_w3 = np.random.uniform(-0.1,0.1,(hid_layer_size, out_layer_size))

	hid_b3 = np.ones(hid_layer_size)
	out_b3 = np.ones(out_layer_size)

	gradient_descent(train_images, convert_labels_to_one_hot(train_labels[0]), hid_w3, out_w3, hid_b3, out_b3, iteration_length, r, lr, batch_size)
	train_data_accuracy(train_images, train_labels, hid_w3, hid_b3, out_w3, out_b3)
	test_data_accuracy(test_images, test_labels, hid_w3, hid_b3, out_w3, out_b3)


def test_the_effect_of_different_normalizations(train_images, train_labels, test_images,test_labels):
	r = 0.001
	lr = 0.00001
	batch_size = 60000
	iteration_length = 500
	hid_layer_size = 150

	hid_w = np.random.uniform(-0.1,0.1,(in_layer_size, hid_layer_size))
	out_w = np.random.uniform(-0.1,0.1,(hid_layer_size, out_layer_size))

	hid_b = np.ones(hid_layer_size)
	out_b = np.ones(out_layer_size)

	norm_1_train = mean_stdeviation_normalization(train_images)
	norm_1_test = mean_stdeviation_normalization(test_images)

	gradient_descent(norm_1_train, convert_labels_to_one_hot(train_labels[0]), hid_w, out_w, hid_b, out_b, iteration_length, r, lr, batch_size)
	train_data_accuracy(norm_1_train, train_labels, hid_w, hid_b, out_w, out_b)
	test_data_accuracy(norm_1_test, test_labels, hid_w, hid_b, out_w, out_b)

	hid_w2 = np.random.uniform(-0.1,0.1,(in_layer_size, hid_layer_size))
	out_w2 = np.random.uniform(-0.1,0.1,(hid_layer_size, out_layer_size))

	hid_b2 = np.ones(hid_layer_size)
	out_b2 = np.ones(out_layer_size)

	norm_2_train = max_min_normalization(train_images)
	norm_2_test = max_min_normalization(test_images)

	gradient_descent(norm_2_train, convert_labels_to_one_hot(train_labels[0]), hid_w2, out_w2, hid_b2, out_b2, iteration_length, r, lr, batch_size)
	train_data_accuracy(norm_2_train, train_labels, hid_w2, hid_b2, out_w2, out_b2)
	test_data_accuracy(norm_2_test, test_labels, hid_w2, hid_b2, out_w2, out_b2)

def test_the_effect_of_tanh_activation_function(train_images, train_labels, test_images,test_labels):
	r = 0.001
	lr = 0.00001
	batch_size = 60000
	iteration_length = 500
	hid_layer_size = 150

	hid_w = np.random.uniform(-0.1,0.1,(in_layer_size, hid_layer_size))
	out_w = np.random.uniform(-0.1,0.1,(hid_layer_size, out_layer_size))

	hid_b = np.ones(hid_layer_size)
	out_b = np.ones(out_layer_size)

	norm_1_train = mean_stdeviation_normalization(train_images)
	norm_1_test = mean_stdeviation_normalization(test_images)

	gradient_descent(norm_1_train, convert_labels_to_one_hot(train_labels[0]), hid_w, out_w, hid_b, out_b, iteration_length, r, lr, batch_size)

	dataset= np.array(norm_1_train)
	result = np.array(convert_labels_to_one_hot(train_labels[0]))
	hid_amount = dataset.dot(hid_w) + hid_b
	hid_amount_active = tanh(hid_amount)
	out_amount = hid_amount_active.dot(out_w) + out_b
	out_amount_active = (out_amount)
	cnt = 0
	for i in  range (len(result)):
		if (np.argmax(result[i]) == np.argmax(out_amount_active[i])):
			cnt += 1
	print("Training Acc= ", cnt/len(result) * 100, "%")


	dataset2= np.array(norm_1_test)
	result2 = np.array(convert_labels_to_one_hot(test_labels[0]))

	hid_amount = dataset2.dot(hid_w) + hid_b
	hid_amount_active = tanh(hid_amount)
	out_amount = hid_amount_active.dot(out_w) + out_b
	out_amount_active = (out_amount)
	cnt = 0
	for i in  range (len(result2)):
		if (np.argmax(result2[i]) == np.argmax(out_amount_active[i])):
			cnt += 1
	print("New Data Acc= ", cnt/len(result2) * 100, "%")

def test_the_effect_of_linear_activation_function(train_images, train_labels, test_images,test_labels):
	r = 0.001
	lr = 0.00001
	batch_size = 60000
	iteration_length = 500
	hid_layer_size = 150

	hid_w = np.random.uniform(-0.1,0.1,(in_layer_size, hid_layer_size))
	out_w = np.random.uniform(-0.1,0.1,(hid_layer_size, out_layer_size))

	hid_b = np.ones(hid_layer_size)
	out_b = np.ones(out_layer_size)

	norm_1_train = mean_stdeviation_normalization(train_images)
	norm_1_test = mean_stdeviation_normalization(test_images)

	gradient_descent(norm_1_train, convert_labels_to_one_hot(train_labels[0]), hid_w, out_w, hid_b, out_b, iteration_length, r, lr, batch_size)

	dataset= np.array(norm_1_train)
	result = np.array(convert_labels_to_one_hot(train_labels[0]))
	hid_amount = dataset.dot(hid_w) + hid_b
	hid_amount_active = linear(hid_amount)
	out_amount = hid_amount_active.dot(out_w) + out_b
	out_amount_active = (out_amount)
	cnt = 0
	for i in  range (len(result)):
		if (np.argmax(result[i]) == np.argmax(out_amount_active[i])):
			cnt += 1
	print("Training Acc= ", cnt/len(result) * 100, "%")


	dataset2= np.array(norm_1_test)
	result2 = np.array(convert_labels_to_one_hot(test_labels[0]))

	hid_amount = dataset2.dot(hid_w) + hid_b
	hid_amount_active = linear(hid_amount)
	out_amount = hid_amount_active.dot(out_w) + out_b
	out_amount_active = (out_amount)
	cnt = 0
	for i in  range (len(result2)):
		if (np.argmax(result2[i]) == np.argmax(out_amount_active[i])):
			cnt += 1
	print("New Data Acc= ", cnt/len(result2) * 100, "%")

def test_the_effect_of_binary_output_format(train_images, train_labels, test_images,test_labels):
	r = 0.001
	lr = 0.00001
	batch_size = 60000
	iteration_length = 500
	hid_layer_size = 150

	hid_w = np.random.uniform(-0.1,0.1,(in_layer_size, hid_layer_size))
	out_w = np.random.uniform(-0.1,0.1,(hid_layer_size, out_layer_size_binary_format))

	hid_b = np.ones(hid_layer_size)
	out_b = np.ones(out_layer_size_binary_format)

	norm_1_train = mean_stdeviation_normalization(train_images)
	norm_1_test = mean_stdeviation_normalization(test_images)

	gradient_descent(norm_1_train, convert_labels_to_binary(train_labels[0]), hid_w, out_w, hid_b, out_b, iteration_length, r, lr, batch_size)
	train_data_accuracy_with_binary_format(norm_1_train, train_labels, hid_w, hid_b, out_w, out_b)
	test_data_accuracy_with_binary_format(norm_1_test, test_labels, hid_w, hid_b, out_w, out_b)

def main():
	train_images, train_labels, test_images,test_labels = read_the_data()
	# test_the_effect_of_different_epochs(train_images, train_labels, test_images,test_labels)
	# test_the_effect_of_different_learning_rates(train_images, train_labels, test_images,test_labels)
	# test_the_effect_of_different_hidden_layer_size(train_images, train_labels, test_images,test_labels)
	# test_the_effect_of_different_momentum_values(train_images, train_labels, test_images,test_labels)
	# test_the_effect_of_different_initial_values(train_images, train_labels, test_images,test_labels)
	# test_the_effect_of_different_batch_size(train_images, train_labels, test_images,test_labels)
	# test_the_effect_of_different_normalizations(train_images, train_labels, test_images,test_labels)
	test_the_effect_of_tanh_activation_function(train_images, train_labels, test_images,test_labels)
	# test_the_effect_of_linear_activation_function(train_images, train_labels, test_images,test_labels)
	# test_the_effect_of_binary_output_format(train_images, train_labels, test_images,test_labels)




import sys 
import math
import numpy as np

def mean_stdeviation_normalization(x):
	matrix = np.array(x, dtype=float)
	mean = np.mean(x, axis=0)
	var = np.var(x, axis=0)
	for pixelInd in range(len(var)):
		if var[pixelInd]==0:
			var[pixelInd] = 0.00001
	for img in x:
		img = (img-mean)/np.sqrt(var)
	return x

def max_min_normalization(x):
	matrix = np.array(x)
	mins = np.min(x, axis=0)
	maxs = np.max(x, axis=0)
	dnum = np.array(maxs - mins, dtype=float)
	for pixelInd in range(len(dnum)):
		if dnum[pixelInd]==0:
			dnum[pixelInd] = 0.0001
	for img in x:
		img = (img-mins)/dnum
	return x


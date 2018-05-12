from general import *
from neural_network import *


INPUT_FILE = "input.txt"


def getImageFileNames(dir_name):
	allImageFiles = getAllImageFileNamesInThisDir(dir_name)
	return (allImageFiles, dir_name)

def whatLetterIsIt(everySingleImage):
	nn = NeuralNetwork(everySingleImage)
	#nn.get_output() TODO: print the result.

def getAllPaths():
	allPaths = []
	f = open(INPUT_FILE)
	for line in f:
		line = line.rstrip()
		allPaths.append(line)
	return allPaths

def getAllImages(allPaths):
	allImages = []
	images = []
	for path in allPaths:
		images , _ = getImageFileNames(path)
		# print(images)
		for imageFile in images:
			imgPath = path +"/"+ imageFile
			imgContent = readImageFile(imgPath)
			if not imgContent==[]:
				# print(path[-1])
				allImages.append((path[-1], imgContent))
	return allImages

if __name__ == '__main__':
	print("Getting path names")
	allPaths = getAllPaths()
	print("Getting all image files")
	everySingleImage = getAllImages(allPaths)
	print("Number of images: {}".format(len(everySingleImage)))
	# print(everySingleImage[0])
	whatLetterIsIt(everySingleImage)
	
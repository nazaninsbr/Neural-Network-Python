from general import *
from neural_network import *


INPUT_FILE = "input.txt"


def getImageFileNames(dir_name):
	allImageFiles = getAllImageFileNamesInThisDir(dir_name)
	return (allImageFiles, dir_name)

def whatLetterIsIt(dirPath):
	allImageFiles, dir_name = getImageFileNames(dirPath)
	print("Working on: "+dirPath)
	for imageFile in allImageFiles:
		path = dir_name +"/"+ imageFile
		img = readImageFile(path)
		if(img==[]):
			print("<---- "+imageFile+" was read as an empty file ---->")
			continue
		nn = NeuralNetwork(img)
		# print("Image Name: " +imageFile+ " is "+nn.getOutput())

def getAllPaths():
	allPaths = []
	f = open(INPUT_FILE)
	for line in f:
		line = line.rstrip()
		allPaths.append(line)
	return allPaths

if __name__ == '__main__':
	allPaths = getAllPaths()
	for dirPath in allPaths:
		whatLetterIsIt(dirPath)
	
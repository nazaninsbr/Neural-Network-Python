import matplotlib.image as mpimg
from os import listdir
from os.path import isfile, join
import imghdr

def readOtherFormats(file_name):
	img = open(file_name, mode='r+')
	lines = []
	lines = img.readlines()
	# for line in img:
	#     lines.append(line)
	img.close()
	return lines

def readImageFile(file_name):
	if (imghdr.what(file_name)=='png'):
		img= mpimg.imread(file_name)
		# print(img[0][0])
		return img
	return readOtherFormats(file_name)


def getAllImageFileNamesInThisDir(directory_name):
	onlyfiles = [f for f in listdir(directory_name) if (isfile(join(directory_name, f)) and any(word in f for word in ['.png', '.JPG', '.jpg', '.jpeg']))]
	return onlyfiles

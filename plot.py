import igraph as ig

class NetworkShape:
	def __init__(self):
		self.layerOne = 0 
		self.layerTwo = 0
		self.layerThree = 0

	def setLayerNeuronCount(self, lay1=28*28, lay2 = 2*28, lay3 = 10):
		self.layerOne = lay1
		self.layerTwo = lay2
		self.layerThree = lay3

	def drawImage(self):
		N = self.layerThree + self.layerTwo + self.layerOne
		Edges = [(i , j) for i in range(self.layerOne) for j in range(self.layerTwo)]
		Edges23 = [(i , j) for i in range(self.layerTwo) for j in range(self.layerThree)]
		Edges.extend(Edges23)
		G=ig.Graph(Edges, directed=False)
		layt=G.layout('kk', dim=3)


def createNetworkShape():
	nn = NetworkShape()
	nn.setLayerNeuronCount()
	nn.drawImage()



def mainFunc():
	createNetworkShape()

if __name__ == '__main__':
	mainFunc()
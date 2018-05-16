import networkx as nx
import matplotlib.pyplot as plt

class NetworkShape:
	def __init__(self):
		self.layerOne = 0 
		self.layerTwo = 0
		self.layerThree = 0
		self.G = nx.Graph()

	def setLayerNeuronCount(self, lay1=28*28, lay2 = 2*28, lay3 = 10):
		self.layerOne = lay1
		self.layerTwo = lay2
		self.layerThree = lay3

	def drawImage(self):
		edge=[(u,v) for (u,v,d) in self.G.edges(data=True)]

		pos=nx.spring_layout(self.G)

		nx.draw_networkx_nodes(self.G,pos,node_size=500)
		nx.draw_networkx_edges(self.G,pos,edgelist=edge,
					width=0.5)

		labels = nx.get_edge_attributes(self.G,'weight')
		edge_labels=dict([((u,v,),d['weight']) for u,v,d in self.G.edges(data=True)])
		nx.draw_networkx_edge_labels(self.G,pos,edge_labels=edge_labels)

		nx.draw_networkx_labels(self.G,pos,font_size=5,font_family='sans-serif')

		plt.axis('off')
		plt.savefig("neural_net.pdf") # save as pdf
		plt.show()

	def createNodesEdges(self, syn1, syn2):
		for x in range(self.layerOne):
			name = "in "+str(x)
			self.G.add_node(name)

		for x in range(self.layerTwo):
			name = "hid "+str(x)
			self.G.add_node(name)

		for x in range(self.layerThree):
			name = "out "+str(x)
			self.G.add_node(name)

		for x in range(self.layerOne):
			for y in range(self.layerTwo):
				lx = "in "+str(x)
				ly = "hid "+str(y)
				self.G.add_edge(lx, ly,color='r', weight=syn1[x+y])

		for x in range(self.layerTwo):
			for y in range(self.layerThree):
				lx = "hid "+str(x)
				ly = "out "+str(y)
				self.G.add_edge(lx, ly, color='b', weight=syn1[x+y])





def createNetworkShape(syn1, syn2):
	print("Plot the Network")
	nn = NetworkShape()
	nn.setLayerNeuronCount()
	nn.createNodesEdges(syn1, syn2)
	nn.drawImage()



def mainFunc(syn1, syn2):
	createNetworkShape(syn1, syn2)

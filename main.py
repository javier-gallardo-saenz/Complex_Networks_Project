import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
from generate_graphs import *
from generate_opinions import *
from graph_statistics import *
from utils import *
from graph_inference import *

comms = [50] * 4  #communities
intra_degree_seq = [15] * sum(comms)
inter_degree_seq = [1] * sum(comms)
G = generate_hierarchical_configuration_model(intra_degree_seq, inter_degree_seq, comms)

# ----------------------------------------------------
# Opinion generation
# ----------------------------------------------------
opinion_dist = OpinionDistribution(G) # create instance of class OpinionDistribution with graph G
opinion_dist.initialize_opinions(states=[-1, 0, 1], probabilities=[0.4, 0.2, 0.4], label='opinion')
opinion_dist.basic_opinion_generator(label='opinion', num_steps=100)

# ----------------------------------------------------
# Saving graphs?
# ----------------------------------------------------

# ----------------------------------------------------
# Opinion inference
# ----------------------------------------------------
graph_inf = GraphInference(G)  # create instance of class GraphInference with graph G, now we can play with it
v = random.choice(list(graph_inf.graph.nodes()))  # choose random node
r = 1  # radius of the known ball
graph_inf.discrete_majority_voting(node=v, radius=r, label='opinion')
print(graph_inf.cache)
#NOTE: label is the name of the node feature we are going to do inference over





#cmap = plt.get_cmap("viridis")

# Create a color mapping for the discrete 'opinion' values
opinion_colors = {-1: 'red', 0: 'blue', 1: 'green'}
# Get the 'opinion' values and map them to colors
node_colors = [opinion_colors[G.nodes[node]['opinion']] for node in G.nodes()]

nx.draw(G, with_labels=True, node_color=node_colors, edge_color="gray", node_size=500, font_size=4)
plt.show()
#print("Resulting Graph")
#print(G.edges())

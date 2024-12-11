import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
from generate_graphs import *
from graph_statistics import *



comms = [50] * 4  #communities
intra_degree_seq = [15] * sum(comms)
inter_degree_seq = [1] * sum(comms)

#hierarchical configuration model test run
G = generate_hierarchical_configuration_model(intra_degree_seq, inter_degree_seq, comms)
nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=16)
plt.show()
print("Resulting Graph")
print(G.edges())

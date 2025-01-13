import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

from utils import *
from generate_graphs import *
from generate_opinions import *
from utils import *
from graph_inference import *
import math

num_nodes = 1000
size_ini = 100
n_edges = 1
delta = 0.01
labels_prob = {1: 0.4, 0: 0.2, -1: 0.4}
# gamma = 2.5 #power law
# mean = 10
# var = 9
# deg_seq = []
# for n in range(num_nodes):
#     deg_seq += [random.choices(
#         population=range(num_nodes//2),
#         weights=[1/(math.sqrt(2*math.pi*var))*(math.e)**(-(k-mean)**2/2*var) for k in range(num_nodes//2)],
#         k=1
#     )[0]]

#deg_seq = generate_power_law_degree_sequence(num_nodes, gamma=gamma, k_min=2)
G = preferential_attachment_with_colors(num_nodes=num_nodes, num_edges=1, labels=labels_prob,
                                            size_init_graph=size_ini, label='opinion', delta=delta)
# opinion_dist = OpinionDistribution(G)  # create instance of class OpinionDistribution with graph G
# opinion_dist.initialize_opinions(states=[-1, 0, 1], probabilities=[1/3, 1/3, 1/3], label='opinion')
# opinion_dist.basic_opinion_generator(label='opinion', num_iterations=5000)
# opinion_dist.opinion_generator_majority_biased_voter_model(label='opinion', num_iterations=5000, delta=0.1)
# opinion_dist.opinion_generator_discrete_label_propagation(label='opinion', num_iterations=10000)

opinion_colors = {-1: 'red', 0: 'blue', 1: 'green'}
# Get the 'opinion' values and map them to colors
node_colors = [opinion_colors[G.nodes[node]['opinion']] for node in G.nodes()]
nx.draw(G, with_labels=False, node_color=node_colors, edge_color="gray", node_size=10, font_size=4)
plt.show()
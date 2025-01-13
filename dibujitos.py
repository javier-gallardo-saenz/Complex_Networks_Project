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
num_nodes_community = 100
num_comm = 10
sizes = [num_nodes_community] * num_comm
size_ini = 100
n_edges = 1
prob_intra = 0.5
prob_inter= 0.005
delta = 0.01
labels_prob = {1: 0.4, 0: 0.2, -1: 0.4}
gamma = 2.5 #power law
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
global_deg_seq = generate_power_law_degree_sequence(num_nodes, gamma=gamma, k_min=5)
out_deg = [0] * num_nodes
in_deg = [0] * num_nodes
for i, deg in enumerate(global_deg_seq):
    if (deg - deg//10)%2 == 0:
        in_deg[i] = deg - deg//10
        out_deg[i] = deg//10
    else:
        in_deg[i] = deg - deg//10 + 1
        out_deg[i] = deg//10 - 1
if sum(out_deg) % 2 != 0:
    out_deg[0] += 1
G = generate_hierarchical_configuration_model(ext_degree_sequence=out_deg,
                                                  in_degree_sequence=in_deg,
                                                  community_sizes=sizes)
#G = generate_sbm(sizes_sbm=sizes, p_inter=prob_inter, p_intra=prob_intra)
opinion_dist = OpinionDistribution(G)  # create instance of class OpinionDistribution with graph G
opinion_dist.initialize_opinions(states=[-1, 0, 1], probabilities=[1/3, 1/3, 1/3], label='opinion')
#opinion_dist.basic_opinion_generator(label='opinion', num_iterations=10000)
#opinion_dist.opinion_generator_majority_biased_voter_model(label='opinion', num_iterations=10000, delta=0.1)
opinion_dist.opinion_generator_discrete_label_propagation(label='opinion', num_iterations=10000)

opinion_colors = {-1: 'red', 0: 'blue', 1: 'green'}
# Get the 'opinion' values and map them to colors
node_colors = [opinion_colors[G.nodes[node]['opinion']] for node in G.nodes()]
nx.draw(G, with_labels=False, node_color=node_colors, edge_color="gray", node_size=10, font_size=4)
plt.show()
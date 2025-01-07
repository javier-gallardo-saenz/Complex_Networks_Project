import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
from generate_graphs import *
from generate_opinions import *
from utils import *
from graph_inference import *

num_nodes = 500
num_comm = 4
comms = [num_nodes] * num_comm  #communities
p = 1
intra_degree_seq = [20] * sum(comms)
inter_degree_seq = np.random.choice([0, 1], size=sum(comms), p=[1-p, p])
if sum(inter_degree_seq) % 2 != 0:
    if inter_degree_seq[-1] == 0:
        inter_degree_seq[-1] = 1
    else:
        inter_degree_seq[-1] = 0

G = generate_hierarchical_configuration_model(intra_degree_seq, inter_degree_seq, comms)
#v = random.choice(G) # choose one single node
v = random.sample(list(G.nodes()), 50)  # choose a random set of nodes
r_values = [1]  # radius of the known ball

# ----------------------------------------------------
# Opinion generation
# ----------------------------------------------------
opinion_dist = OpinionDistribution(G)  # create instance of class OpinionDistribution with graph G
opinion_dist.initialize_opinions(states=[-1, 0, 1], probabilities=[0.4, 0.2, 0.4], label='opinion')
opinion_dist.basic_opinion_generator(label='opinion', num_steps=100000)
proportion_of_labels(nodes_per_comm=num_nodes, num_communities=num_comm, Graph=G, label='opinion')

# ----------------------------------------------------
# Saving graphs?
# ----------------------------------------------------

# ----------------------------------------------------
# Opinion inference
# ----------------------------------------------------
# create instance of class GraphInference with graph G, now we can play with it
graph_inf = GraphInference(opinion_dist.graph)
graph_inf.which_inference_methods()  # shows available inference methods
methods = {'dlp'}
results_dmv = graph_inf.do_inference(node_set=v, radius_values=r_values, methods=methods, label='opinion',
                                                          count_results=2, clear_results=False, num_iterations=1)

# for method in methods:
#     for r in r_values:
#         print(f"The fraction of correct guesses with r = {r} and method {method}"
#               f" was {results_dmv[method][r]['success'] / results_dmv[method][r]['visited_nodes']}.")
#         print(f"The average distance of the inferred opinion to the true opinion with r = {r} and method {method}"
#               f" was {results_dmv[method][r]['acc_dist'] / results_dmv[method][r]['visited_nodes']}.")

for method in methods:
    for r in r_values:
        aux = get_all_stats(results_dmv[method][r]['inferred'], results_dmv[method][r]['true'], [-1, 0, 1])
        print(f"The stats for method {method} and r = {r} are:")
        print(aux)


# # get true labels of the boundary of the ball
# true_labels = graph_inf.get_true_label(node=v, radius=r, label='opinion')
# # get inferred labels of the boundary of the ball
# inferred_labels = graph_inf.get_inferred_label(node=v, radius=r, method_name='dmv', label='opinion')
# print(true_labels)
# print(inferred_labels)
# #NOTE: label is the name of the node feature we are going to do inference over


# ----------------------------------------------------
# Plots
# ----------------------------------------------------
#cmap = plt.get_cmap("viridis")

# Create a color mapping for the discrete 'opinion' values
opinion_colors = {-1: 'red', -0.5: 'pink', 0: 'blue', 0.5: 'yellow', 1: 'green'}
# Get the 'opinion' values and map them to colors
node_colors = [opinion_colors[G.nodes[node]['opinion']] for node in G.nodes()]

nx.draw(G, with_labels=True, node_color=node_colors, edge_color="gray", node_size=500, font_size=4)
plt.show()
#print("Resulting Graph")
#print(G.edges())

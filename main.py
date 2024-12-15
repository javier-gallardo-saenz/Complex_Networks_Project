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
v = list(G.nodes())  # choose all nodes
r_values = [1, 2, 3]  # radius of the known ball

# ----------------------------------------------------
# Opinion generation
# ----------------------------------------------------
opinion_dist = OpinionDistribution(G)  # create instance of class OpinionDistribution with graph G
opinion_dist.initialize_opinions(states=[-1, 0, 1], probabilities=[0.4, 0.2, 0.4], label='opinion')
opinion_dist.basic_opinion_generator(label='opinion', num_steps=10000)

# ----------------------------------------------------
# Saving graphs?
# ----------------------------------------------------

# ----------------------------------------------------
# Opinion inference
# ----------------------------------------------------
# create instance of class GraphInference with graph G, now we can play with it
graph_inf = GraphInference(opinion_dist.graph)
graph_inf.which_inference_methods()  # shows available inference methods
methods = {'dvm'}
results_dmv = graph_inf.do_inference(node_set=v, radius_values=r_values, methods=methods, label='opinion',
                                                          count_results=True, clear_results=False, num_iterations=1000)

for method in methods:
    for r in r_values:
        print(f"The fraction of correct guesses with r = {r} and method {method}"
              f" was {results_dmv[method][r][0] / results_dmv[method][r][2]}.")
        print(f"The average distance of the inferred opinion to the true opinion with r = {r} and method {method}"
              f" was {results_dmv[method][r][1] / results_dmv[method][r][2]}.")


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
opinion_colors = {-1: 'red', 0: 'blue', 1: 'green'}
# Get the 'opinion' values and map them to colors
node_colors = [opinion_colors[G.nodes[node]['opinion']] for node in G.nodes()]

nx.draw(G, with_labels=True, node_color=node_colors, edge_color="gray", node_size=500, font_size=4)
plt.show()
#print("Resulting Graph")
#print(G.edges())

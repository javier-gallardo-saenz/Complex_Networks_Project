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

num_nodes = 500
num_edges = 8
G = generate_albert_barabasi_graph(n=num_nodes, m=num_edges)
#v = random.choice(G) # choose one single node
v = random.sample(list(G.nodes()), len(list(G.nodes()))//3)  # choose a random set of nodes
r_values = [1]  # radius of the known ball

# ----------------------------------------------------
# Opinion generation
# ----------------------------------------------------
opinion_dist = OpinionDistribution(G)  # create instance of class OpinionDistribution with graph G
opinion_dist.initialize_opinions(states=[-1, 0, 1], probabilities=[0.4, 0.2, 0.4], label='opinion')
opinion_dist.basic_opinion_generator(label='opinion', num_steps=10000)
# proportion_of_labels(nodes_per_comm=num_nodes, num_communities=num_comm, Graph=G, label='opinion')

# ----------------------------------------------------
# Saving graphs?
# ----------------------------------------------------

# ----------------------------------------------------
# Opinion inference
# ----------------------------------------------------
# create instance of class GraphInference with graph G, now we can play with it
graph_inf = GraphInference(opinion_dist.graph)
graph_inf.which_inference_methods()  # shows available inference methods
methods = {'dmv', 'dwmv', 'dvm', 'dlp'}
results_dmv = graph_inf.do_inference(node_set=v, radius_values=r_values, methods=methods, label='opinion',
                                                          count_results=2, clear_results=False, num_iterations=1)


# for method in methods:
#    for r in r_values:
#        print(f"The fraction of correct guesses with r = {r} and method {method}"
#              f" was {results_dmv[method][r][0] / results_dmv[method][r][2]}.")
#        print(f"The average distance of the inferred opinion to the true opinion with r = {r} and method {method}"
#              f" was {results_dmv[method][r][1] / results_dmv[method][r][2]}.")
        
        
num_iterations = 10
avg_correct_guesses = {}
avg_distance = {}
for n in range(num_iterations):
      graph_inf = GraphInference(opinion_dist.graph)
      graph_inf.which_inference_methods()  # shows available inference methods
      methods = {'dmv', 'dwmv', 'dvm', 'dlp'}
      results_dmv = graph_inf.do_inference(node_set=v, radius_values=r_values, methods=methods, label='opinion',
                                                            count_results=True, clear_results=False, num_iterations=1)
      for method in methods:
            for r in r_values:
                  if method not in avg_correct_guesses.keys():
                        avg_correct_guesses[method] = {r:results_dmv[method][r][0] / results_dmv[method][r][2]}
                  else:
                       avg_correct_guesses[method][r]+=results_dmv[method][r][0] / results_dmv[method][r][2] 
                  if method not in avg_distance.keys():
                        avg_distance[method] = {r:results_dmv[method][r][1] / results_dmv[method][r][2]}
                  else:
                        avg_distance[method][r]+=results_dmv[method][r][1] / results_dmv[method][r][2]

print(f"Results for Barabási-Albert with {num_nodes} nodes and {num_edges} edges over 1/3 of the node set.")
for method in methods:
      for r in r_values:
            print(f"The average fraction of correct guesses with r = {r} and method {method} of {num_iterations} iterations"
                  f" was {avg_correct_guesses[method][r]/num_iterations}.")
            print(f"The average distance of the inferred opinion to the true opinion with r = {r} and method {method}"
                  f" of {num_iterations} iterations was {avg_distance[method][r]/num_iterations}.")

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
# opinion_colors = {-1: 'red', -0.5: 'pink', 0: 'blue', 0.5: 'yellow', 1: 'green'}
# Get the 'opinion' values and map them to colors
# node_colors = [opinion_colors[G.nodes[node]['opinion']] for node in G.nodes()]

#nx.draw(G, with_labels=True, node_color=node_colors, edge_color="gray", node_size=500, font_size=4)
#plt.show()
#print("Resulting Graph")
#print(G.edges())

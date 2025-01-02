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

n_nodes = 1000
size_ini = 100
n_edges = 3
labels_prob = {1:1/3, 0:1/3, -1:1/3}
r_values = [1]  # radius of the known ball

num_iterations = 10
avg_correct_guesses = {}
avg_distance = {}
for n in range(num_iterations):
    G = preferential_attachment_with_colors(num_nodes=n_nodes, num_edges=n_edges, labels=labels_prob, 
                                            size_init_graph=size_ini, label='opinion', delta=0.1)
    v = random.sample(list(G.nodes()), len(list(G.nodes()))//3)  # choose a random set of nodes
    graph_inf = GraphInference(G)
    graph_inf.which_inference_methods()  # shows available inference methods
    methods = {'dmv', 'dwmv', 'dvm', 'dlp'}
    results_dmv = graph_inf.do_inference(node_set=v, radius_values=r_values, methods=methods, label='opinion',
                                                        count_results=2, clear_results=False, num_iterations=1)
    
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

print(f"Results for Pref. Attach. with Colors with {n_nodes} nodes, {n_edges} num_edges,"
      f" {size_ini} initial size, and {labels_prob} labels probabilities over 1/3 of the node set.")
for method in methods:
      for r in r_values:
            print(f"The average fraction of correct guesses with r = {r} and method {method} of {num_iterations} iterations"
                  f" was {avg_correct_guesses[method][r]/num_iterations}.")
            print(f"The average distance of the inferred opinion to the true opinion with r = {r} and method {method}"
                  f" of {num_iterations} iterations was {avg_distance[method][r]/num_iterations}.")
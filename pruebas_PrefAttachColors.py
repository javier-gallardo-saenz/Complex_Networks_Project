import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
from generate_graphs import *
from generate_opinions import *
from utils import *
from graph_inference import *

n_nodes = 1000
size_ini = 100
n_edges = 3
labels_prob = {1:1/3, 0:1/3, -1:1/3}
r_values = [1]  # radius of the known ball

num_iterations = 10
avg_aux = {}
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
        if method not in avg_aux.keys():
           avg_aux[method] = {}
        for r in r_values:
            aux = get_all_stats(results_dmv[method][r]['inferred'], results_dmv[method][r]['true'], labels=[-1, 0, 1])
            print(f"The stats for method {method} and r = {r} are:")
            print(aux)
            for key in aux.keys():
                if key not in avg_aux[method].keys():
                    avg_aux[method][key] = aux[key]/num_iterations
                else:
                    avg_aux[method][key] += aux[key]/num_iterations

print(f"Results for Pref. Attach. with Colors with {n_nodes} nodes, {n_edges} num_edges,"
      f" {size_ini} initial size, and {labels_prob} labels probabilities over 1/3 of the node set.")
for key in avg_aux.keys():
    print(f"\n{key} :")
    for key2 in avg_aux[key].keys():
        print(f"{key2} : {avg_aux[key][key2]}")
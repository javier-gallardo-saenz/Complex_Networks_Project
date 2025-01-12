import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
from generate_graphs import *
from generate_opinions import *
from utils import *
from graph_inference import *

num_nodes = 10000
size_ini = 100
n_edges = 10
delta = 0.01
labels_prob = {1: 0.4, 0: 0.2, -1: 0.4}
r_values = [0, 1]  # radius of the known ball
methods = {'dmv', 'dwmv', 'dvm', 'dlp'}
total_results = {method: {r: {'inferred': [], 'true': []} for r in r_values} for method in methods}
total_weighted_stats = {method: {r: {} for r in r_values} for method in methods}
selected_nodes_per_graph = num_nodes//100

num_iterations = 50
avg_aux = {}
for n in range(num_iterations):
    G = preferential_attachment_with_colors(num_nodes=num_nodes, num_edges=n_edges, labels=labels_prob,
                                            size_init_graph=size_ini, label='opinion', delta=delta)
    v = random.sample(list(G.nodes()), 10)  # choose a random set of nodes
    graph_inf = GraphInference(G)
    graph_inf.which_inference_methods()  # shows available inference methods
    results_dmv = graph_inf.do_inference(node_set=v, radius_values=r_values, methods=methods, label='opinion',
                                                        count_results=2, clear_results=False, num_iterations=1)
    proportion_of_labels_total(Graph=G, label='opinion')
    for method in methods:
        if method not in avg_aux.keys():
           avg_aux[method] = {}

        for r in r_values:
            total_results[method][r]['inferred'] += results_dmv[method][r]['inferred']
            total_results[method][r]['true'] += results_dmv[method][r]['true']

            if r not in avg_aux[method].keys():
                avg_aux[method][r] = {}

            aux = get_all_stats(results_dmv[method][r]['inferred'], results_dmv[method][r]['true'], labels=[-1, 0, 1])

            for key in aux.keys():
                if key not in avg_aux[method][r].keys():
                    avg_aux[method][r][key] = aux[key]/num_iterations
                else:
                    avg_aux[method][r][key] += aux[key]/num_iterations


for method in methods:
    for r in r_values:
        total_weighted_stats[method][r] = get_all_stats(total_results[method][r]['inferred'],
                                                        total_results[method][r]['true'], labels=[-1, 0, 1])

print(f"Results for Pref. Attach. with Colors with {n_nodes} nodes, {n_edges} num_edges,"
      f" {size_ini} initial size, and {labels_prob} labels probabilities "
      f"over {selected_nodes_per_graph} of the node set.")
for method in avg_aux.keys():
    print(f"\n{method} :")
    for r in avg_aux[method].keys():
        print(f"\t{r} : ")
        for key2 in avg_aux[method][r].keys():
            print(f"\t Iterations averaged {key2} : {avg_aux[method][r][key2]}")
            print(f"\t Inferred nodes averaged {key2} : {total_weighted_stats[method][r][key2]}")
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
# mean = 100
# var = 99
#deg_seq = []
gamma = 1.5 #power law
r_values = [0, 1] 
methods = {'dmv', 'dwmv', 'dvm', 'dlp'}
selected_nodes_per_graph = num_nodes//100
"""
for n in range(num_nodes):
    deg_seq += [random.choices(
        population=range(num_nodes//2),
        weights=[1/(math.sqrt(2*math.pi*var))*(math.e)**(-(k-mean)**2/2*var) for k in range(num_nodes//2)],
        k=1
    )[0]]
"""
total_results = {method: {r: {'inferred': [], 'true': []} for r in r_values} for method in methods}
total_weighted_stats = {method: {r: {} for r in r_values} for method in methods}

num_iterations = 1
avg_aux = {}
for n in range(num_iterations):
    deg_seq = generate_power_law_degree_sequence(num_nodes, gamma=gamma, k_min=5)
    G = generate_configuration_model(degree_sequence=deg_seq)
    v = random.sample(list(G.nodes()), selected_nodes_per_graph)  # choose a random set of nodes
    opinion_dist = OpinionDistribution(G)  # create instance of class OpinionDistribution with graph G
    opinion_dist.initialize_opinions(states=[-1, 0, 1], probabilities=[1/3, 1/3, 1/3], label='opinion')
    #opinion_dist.basic_opinion_generator(label='opinion', num_iterations=10000)
    #opinion_dist.opinion_generator_majority_biased_voter_model(label='opinion', num_iterations=10000, delta=0.1)
    opinion_dist.opinion_generator_discrete_label_propagation(label='opinion', num_iterations=10000)
    graph_inf = GraphInference(opinion_dist.graph)
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

print(f"Results for the Configuration Model with {num_nodes} nodes and a degree sequence with power law {gamma}"
      f"over {num_iterations} CM graphs.")
print(f"Inference was done independently on the frontier of balls centered on "
      f"{selected_nodes_per_graph} nodes sampled randomly")

for method in avg_aux.keys():
    print(f"\n Method {method} :")
    for r in avg_aux[method].keys():
        print(f"\t Radius {r} : ")
        for key2 in avg_aux[method][r].keys():
            print(f"\t Iterations averaged {key2} : {avg_aux[method][r][key2]}")
            print(f"\t Inferred nodes averaged {key2} : {total_weighted_stats[method][r][key2]}")
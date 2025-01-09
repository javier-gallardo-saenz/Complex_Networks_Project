import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
from generate_graphs import *
from generate_opinions import *
from utils import *
from graph_inference import *

num_nodes = 1000
prob_edge = 0.05
r_values = [0, 1]  # radius of the known ball
methods = {'dmv', 'dwmv', 'dvm', 'dlp'}

num_iterations = 50
avg_aux = {}
total_results = {method: {r: {'inferred': [], 'true': []} for r in r_values} for method in methods}
total_weighted_stats = {method: {r: {} for r in r_values} for method in methods}

for n in range(num_iterations):
    G = generate_erdos_renyi_graph(n=num_nodes, p=prob_edge)
    v = random.sample(list(G.nodes()), 10)  # choose a random set of nodes
    opinion_dist = OpinionDistribution(G)  # create instance of class OpinionDistribution with graph G
    opinion_dist.initialize_opinions(states=[-1, 0, 1], probabilities=[0.33, 0.34, 0.33], label='opinion')
    opinion_dist.basic_opinion_generator(label='opinion', num_steps=10000)
    graph_inf = GraphInference(opinion_dist.graph)
    graph_inf.which_inference_methods()  # shows available inference methods

    results_dmv = graph_inf.do_inference(node_set=v, radius_values=r_values, methods=methods, label='opinion',
                                                        count_results=2, clear_results=False, num_iterations=1)



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


print(f"Results for Erdos-Renyi with {num_nodes} nodes and {prob_edge} probability of"
      f" an edge over 1/3 of the node set.")

for method in avg_aux.keys():
    print(f"\n{method} :")
    for r in avg_aux[method].keys():
        print(f"\t{r} : ")
        for key2 in avg_aux[method][r].keys():
            print(f"\t Iterations averaged {key2} : {avg_aux[method][r][key2]}")
            print(f"\t Inferred nodes averaged {key2} : {total_weighted_stats[method][r][key2]}")


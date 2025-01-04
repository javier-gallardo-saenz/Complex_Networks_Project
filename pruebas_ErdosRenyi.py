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
prob_edge = 0.15
r_values = [1]  # radius of the known ball

num_iterations = 10
avg_aux = {}
for n in range(num_iterations):
    G = generate_erdos_renyi_graph(n=num_nodes, p=prob_edge)
    v = random.sample(list(G.nodes()), len(list(G.nodes()))//3)  # choose a random set of nodes
    opinion_dist = OpinionDistribution(G)  # create instance of class OpinionDistribution with graph G
    opinion_dist.initialize_opinions(states=[-1, 0, 1], probabilities=[0.33, 0.34, 0.33], label='opinion')
    opinion_dist.basic_opinion_generator(label='opinion', num_steps=10000)
    graph_inf = GraphInference(opinion_dist.graph)
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

print(f"Results for Erdos-Renyi with {num_nodes} nodes and {prob_edge} probability of"
      f" an edge over 1/3 of the node set.")
for key in avg_aux.keys():
    print(f"\n{key} :")
    for key2 in avg_aux[key].keys():
        print(f"{key2} : {avg_aux[key][key2]}")
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
prob_edge = 0.005
r_values = [0, 1]  # radius of the known ball
methods = {'dmv', 'dwmv', 'dvm', 'dlp'}
selected_nodes_per_graph = 100

num_iterations = 5
avg_aux = {}
total_results = {method: {r: {'inferred': [], 'true': []} for r in r_values} for method in methods}
total_weighted_stats = {method: {r: {} for r in r_values} for method in methods}

for n in range(num_iterations):
    G = generate_erdos_renyi_graph(n=num_nodes, p=prob_edge)
    v = random.sample(list(G.nodes()), selected_nodes_per_graph)  # choose a random set of nodes
    opinion_dist = OpinionDistribution(G)  # create instance of class OpinionDistribution with graph G
    opinion_dist.initialize_opinions(states=[-1, 0, 1], probabilities=[0.4, 0.2, 0.4], label='opinion')
    opinion_dist.basic_opinion_generator(label='opinion', num_iterations=100000)
    #opinion_dist.opinion_generator_majority_biased_voter_model(label='opinion', num_iterations=10000, delta=0.1)
    graph_inf = GraphInference(opinion_dist.graph)
    #graph_inf.which_inference_methods()  # shows available inference methods

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


print(f"Results for Erdos-Renyi with {num_nodes} nodes and {prob_edge} probability of"
      f" an edge over {num_iterations} ER graphs.")
print(f"Inference was done independently on the frontier of balls centered on "
      f"{selected_nodes_per_graph} nodes sampled randomly")

for method in avg_aux.keys():
    print(f"\n Method {method} :")
    for r in avg_aux[method].keys():
        print(f"\t Radius {r} : ")
        for key2 in avg_aux[method][r].keys():
            print(f"\t Iterations averaged {key2} : {avg_aux[method][r][key2]}")
            print(f"\t Inferred nodes averaged {key2} : {total_weighted_stats[method][r][key2]}")


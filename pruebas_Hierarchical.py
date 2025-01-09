import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
from generate_graphs import *
from generate_opinions import *
from utils import *
from graph_inference import *
from visualize_graphs import *

num_nodes = 200
num_comm = 5
comms = [num_nodes] * num_comm  #communities
mean_intra = 100
var_intra = 50
mean_inter = 40
var_inter = 38
#degree_intra = 150
#degree_inter = 5
intra_degree_seq = one_sided_normal_degree_seq(num_nodes=sum(comms), mean=mean_intra, var=var_intra)
for k in range(num_comm):
    if sum(intra_degree_seq[num_nodes*k : num_nodes*(k+1)])%2 != 0:
        intra_degree_seq[num_nodes*k] -= 1
inter_degree_seq = one_sided_normal_degree_seq(num_nodes=sum(comms), mean=mean_inter, var=var_inter)
r_values = [0, 1]  # radius of the known ball
methods = {'dmv', 'dwmv', 'dvm', 'dlp'}
total_results = {method: {r: {'inferred': [], 'true': []} for r in r_values} for method in methods}
total_weighted_stats = {method: {r: {} for r in r_values} for method in methods}

num_iterations = 50
avg_aux = {}
for n in range(num_iterations):
    G = generate_hierarchical_configuration_model(ext_degree_sequence=inter_degree_seq, 
                                                  in_degree_sequence=intra_degree_seq,
                                                  community_sizes=comms)
    v = random.sample(list(G.nodes()), 10)  # choose a random set of nodes
    opinion_dist = OpinionDistribution(G)  # create instance of class OpinionDistribution with graph G
    opinion_dist.initialize_opinions(states=[-1, 0, 1], probabilities=[1/3, 1/3, 1/3], label='opinion')
    # opinion_dist.basic_opinion_generator(label='opinion', num_steps=10000)
    opinion_dist.opinion_generator_majority_biased_voter_model(label='opinion', num_iterations=1000, delta=0.1)
    graph_inf = GraphInference(opinion_dist.graph)
    graph_inf.which_inference_methods()  # shows available inference methods
    methods = {'dmv', 'dwmv', 'dvm', 'dlp'}
    results_dmv = graph_inf.do_inference(node_set=v, radius_values=r_values, methods=methods, label='opinion',
                                                        count_results=2, clear_results=False, num_iterations=1)
    proportion_of_labels(num_communities=num_comm, nodes_per_comm=num_nodes, Graph=G, label='opinion')
    #graph_labels={}
    #for node in G.nodes:
    #    graph_labels[node] = G.nodes[node]['opinion']
    #visualize_graph(G, graph_labels)
    
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

print(f"Results for Hierarchical with {comms} communities, mean {mean_intra}, var {var_intra} degree inter-communities"
      f" and mean {mean_inter}, var {var_inter} degree intra-communities over 10 nodes of the set.")
for method in avg_aux.keys():
    print(f"\n{method} :")
    for r in avg_aux[method].keys():
        print(f"\t{r} : ")
        for key2 in avg_aux[method][r].keys():
            print(f"\t Iterations averaged {key2} : {avg_aux[method][r][key2]}")
            print(f"\t Inferred nodes averaged {key2} : {total_weighted_stats[method][r][key2]}")
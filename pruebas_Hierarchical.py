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

num_nodes = 1000
num_comm = 10
comms = [num_nodes//num_comm] * num_comm  #communities
# mean_intra = 50
# var_intra = 25
# mean_inter = 5
# var_inter = 5
# #degree_intra = 150
# #degree_inter = 5
# intra_degree_seq = one_sided_normal_degree_seq(num_nodes=sum(comms), mean=mean_intra, var=var_intra)
# for k in range(num_comm):
#     if sum(intra_degree_seq[num_nodes*k : num_nodes*(k+1)])%2 != 0:
#         intra_degree_seq[num_nodes*k] -= 1
# inter_degree_seq = one_sided_normal_degree_seq(num_nodes=sum(comms), mean=mean_inter, var=var_inter)
gamma = 2.5
r_values = [0, 1]  # radius of the known ball
methods = {'dmv', 'dwmv', 'dvm', 'dlp'}
num_iterations = 1

for n in range(num_iterations):
    total_results = {method: {r: {'inferred': [], 'true': []} for r in r_values} for method in methods}
    total_weighted_stats = {method: {r: {} for r in r_values} for method in methods}
    avg_aux = {}
    global_deg_seq = generate_power_law_degree_sequence(num_nodes, gamma=gamma, k_min=10)
    out_deg = [0] * num_nodes
    in_deg = [0] * num_nodes
    for i, deg in enumerate(global_deg_seq):
        if (deg - deg//10)%2 == 0:
            in_deg[i] = deg - deg//10
            out_deg[i] = deg//10
        else:
            in_deg[i] = deg - deg//10 + 1
            out_deg[i] = deg//10 - 1
    if sum(out_deg) % 2 != 0:
        out_deg[0] += 1
    # out_deg_skewed = generate_power_law_degree_sequence(num_nodes, gamma=gamma_out, k_min=1)
    # out_deg = [x - 1 for x in out_deg_skewed]
    # in_deg = [
    #     degree
    #     for _ in range(num_comm)
    #     for degree in generate_power_law_degree_sequence(num_nodes//num_comm, gamma=gamma_in, k_min=10)
    # ]
    G = generate_hierarchical_configuration_model(ext_degree_sequence=out_deg,
                                                  in_degree_sequence=in_deg,
                                                  community_sizes=comms)
    v = random.sample(list(G.nodes()), num_nodes//100)  # choose a random set of nodes
    opinion_dist = OpinionDistribution(G)  # create instance of class OpinionDistribution with graph G
    opinion_dist.initialize_opinions(states=[-1, 0, 1], probabilities=[1/3, 1/3, 1/3], label='opinion')
    # opinion_dist.basic_opinion_generator(label='opinion', num_steps=100000)
    opinion_dist.opinion_generator_majority_biased_voter_model(label='opinion', num_iterations=10000, delta=0.1)
    graph_inf = GraphInference(opinion_dist.graph)
    results_dmv = graph_inf.do_inference(node_set=v, radius_values=r_values, methods=methods, label='opinion',
                                                        count_results=2, clear_results=False, num_iterations=1)
    proportion_of_labels(num_communities=num_comm, nodes_per_comm=num_nodes//num_comm, Graph=G, label='opinion')
    #fit_powerlaw(G)
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

print(f"Results for Hierarchical with {num_nodes} communities, {num_comm} communities, "
                  f"power coefficient{gamma}; "
                  f"opinion generator basic")
for method in avg_aux.keys():
    print(f"\n{method} :")
    for r in avg_aux[method].keys():
        print(f"\t{r} : ")
        for key2 in avg_aux[method][r].keys():
            print(f"\t Iterations averaged {key2} : {avg_aux[method][r][key2]}")
            print(f"\t Inferred nodes averaged {key2} : {total_weighted_stats[method][r][key2]}")



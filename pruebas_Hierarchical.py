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
num_comm = 4
comms = [num_nodes] * num_comm  #communities
degree_intra = 60
degree_inter = 1
intra_degree_seq = [degree_intra] * sum(comms)
inter_degree_seq = [degree_inter] * sum(comms)
r_values = [1]  # radius of the known ball

num_iterations = 1
avg_correct_guesses = {}
avg_distance = {}
for n in range(num_iterations):
    G = generate_hierarchical_configuration_model(ext_degree_sequence=inter_degree_seq, 
                                                  in_degree_sequence=intra_degree_seq,
                                                  community_sizes=comms)
    v = random.sample(list(G.nodes()), len(list(G.nodes()))//3)  # choose a random set of nodes
    opinion_dist = OpinionDistribution(G)  # create instance of class OpinionDistribution with graph G
    opinion_dist.initialize_opinions(states=[-1, 0, 1], probabilities=[0.33, 0.33, 0.34], label='opinion')
    opinion_dist.basic_opinion_generator(label='opinion', num_steps=10000)
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
        for r in r_values:
                if method not in avg_correct_guesses.keys():
                    avg_correct_guesses[method] = {r:results_dmv[method][r][0] / results_dmv[method][r][2]}
                else:
                    avg_correct_guesses[method][r]+=results_dmv[method][r][0] / results_dmv[method][r][2] 
                if method not in avg_distance.keys():
                    avg_distance[method] = {r:results_dmv[method][r][1] / results_dmv[method][r][2]}
                else:
                    avg_distance[method][r]+=results_dmv[method][r][1] / results_dmv[method][r][2]

print(f"Results for Hierarchical with {comms} communities, {degree_inter} degree inter-communities"
      f" and {degree_intra} degree intra-communities over 1/3 of the node set.")
for method in methods:
      for r in r_values:
            print(f"The average fraction of correct guesses with r = {r} and method {method} of {num_iterations} iterations"
                  f" was {avg_correct_guesses[method][r]/num_iterations}.")
            print(f"The average distance of the inferred opinion to the true opinion with r = {r} and method {method}"
                  f" of {num_iterations} iterations was {avg_distance[method][r]/num_iterations}.")
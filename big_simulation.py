from generate_graphs import *
from generate_opinions import *
from utils import *
from graph_inference import *

num_nodes_list = [1000, 10000, 50000]
num_comm_list = [5, 100, 200]
num_iterations_list = [50, 5, 1]
power = [1.5, 2.5]
added_edges_per_step = [1, 5, 10]
size_ini = 100
delta = 0.01
labels_prob = {1: 1 / 3, 0: 1 / 3, -1: 1 / 3}
r_values = [0, 1]  # radius of the known ball
methods = {'dmv', 'dwmv', 'dvm', 'dlp'}

for j in range(len(num_nodes_list)):
    num_nodes = num_nodes_list[j]
    num_comm = num_comm_list[j]
    num_iterations = num_iterations_list[j]
    comms = [num_nodes // num_comm] * num_comm  # communities
    selected_nodes_per_graph = num_nodes // 100

    ########
    # HCM  #
    ########
    for gamma in power:
        #######
        #BASIC#
        #######
        total_results = {method: {r: {'inferred': [], 'true': []} for r in r_values} for method in methods}
        total_weighted_stats = {method: {r: {} for r in r_values} for method in methods}
        avg_aux = {}
        for _ in range(num_iterations):
            global_deg_seq = generate_power_law_degree_sequence(num_nodes, gamma=gamma, k_min=10)
            out_deg = [0] * num_nodes
            in_deg = [0] * num_nodes
            for i, deg in enumerate(global_deg_seq):
                if (deg - deg // 10) % 2 == 0:
                    in_deg[i] = deg - deg // 10
                    out_deg[i] = deg // 10
                else:
                    in_deg[i] = deg - deg // 10 + 1
                    out_deg[i] = deg // 10 - 1
            G = generate_hierarchical_configuration_model(ext_degree_sequence=out_deg,
                                                          in_degree_sequence=in_deg,
                                                          community_sizes=comms)
            v = random.sample(list(G.nodes()), selected_nodes_per_graph)  # choose a random set of nodes
            opinion_dist = OpinionDistribution(G)  # create instance of class OpinionDistribution with graph G
            opinion_dist.initialize_opinions(states=[-1, 0, 1], probabilities=[0.4, 0.2, 0.4], label='opinion')
            opinion_dist.basic_opinion_generator(label='opinion', num_iterations=10 * num_nodes)
            graph_inf = GraphInference(opinion_dist.graph)
            results_dmv = graph_inf.do_inference(node_set=v, radius_values=r_values, methods=methods,
                                                 label='opinion',
                                                 count_results=2, clear_results=False, num_iterations=1)
            proportion_of_labels(num_communities=num_comm, nodes_per_comm=num_nodes // num_comm,
                                 Graph=G, label='opinion')

            for method in methods:
                if method not in avg_aux.keys():
                    avg_aux[method] = {}

                for r in r_values:
                    total_results[method][r]['inferred'] += results_dmv[method][r]['inferred']
                    total_results[method][r]['true'] += results_dmv[method][r]['true']

                    if r not in avg_aux[method].keys():
                        avg_aux[method][r] = {}

                    aux = get_all_stats(results_dmv[method][r]['inferred'], results_dmv[method][r]['true'],
                                        labels=[-1, 0, 1])

                    for key in aux.keys():
                        if key not in avg_aux[method][r].keys():
                            avg_aux[method][r][key] = aux[key] / num_iterations
                        else:
                            avg_aux[method][r][key] += aux[key] / num_iterations

        for method in methods:
            for r in r_values:
                total_weighted_stats[method][r] = get_all_stats(total_results[method][r]['inferred'],
                                                                total_results[method][r]['true'], labels=[-1, 0, 1])

        print(f"Results for Hierarchical with {num_nodes} communities, {num_comm} communities, "
              f"power coefficient {gamma}; opinion generator basic")
        for method in avg_aux.keys():
            print(f"\n{method} :")
            for r in avg_aux[method].keys():
                print(f"\t{r} : ")
                for key2 in avg_aux[method][r].keys():
                    print(f"\t Iterations averaged {key2} : {avg_aux[method][r][key2]}")
                    print(f"\t Inferred nodes averaged {key2} : {total_weighted_stats[method][r][key2]}")

        #######
        # MV  #
        #######
        total_results = {method: {r: {'inferred': [], 'true': []} for r in r_values} for method in methods}
        total_weighted_stats = {method: {r: {} for r in r_values} for method in methods}
        avg_aux = {}
        for _ in range(num_iterations):
            global_deg_seq = generate_power_law_degree_sequence(num_nodes, gamma=gamma, k_min=10)
            out_deg = [0] * num_nodes
            in_deg = [0] * num_nodes
            for i, deg in enumerate(global_deg_seq):
                if (deg - deg // 10) % 2 == 0:
                    in_deg[i] = deg - deg // 10
                    out_deg[i] = deg // 10
                else:
                    in_deg[i] = deg - deg // 10 + 1
                    out_deg[i] = deg // 10 - 1
            G = generate_hierarchical_configuration_model(ext_degree_sequence=out_deg,
                                                          in_degree_sequence=in_deg,
                                                          community_sizes=comms)
            v = random.sample(list(G.nodes()), selected_nodes_per_graph)  # choose a random set of nodes
            opinion_dist = OpinionDistribution(G)  # create instance of class OpinionDistribution with graph G
            opinion_dist.initialize_opinions(states=[-1, 0, 1], probabilities=[1 / 3, 1 / 3, 1 / 3],
                                             label='opinion')
            opinion_dist.opinion_generator_majority_biased_voter_model(label='opinion',
                                                                       num_iterations=10 * num_nodes, delta=0.1)
            graph_inf = GraphInference(opinion_dist.graph)
            results_dmv = graph_inf.do_inference(node_set=v, radius_values=r_values, methods=methods,
                                                 label='opinion',
                                                 count_results=2, clear_results=False, num_iterations=1)
            proportion_of_labels(num_communities=num_comm, nodes_per_comm=num_nodes // num_comm,
                                 Graph=G, label='opinion')

            for method in methods:
                if method not in avg_aux.keys():
                    avg_aux[method] = {}

                for r in r_values:
                    total_results[method][r]['inferred'] += results_dmv[method][r]['inferred']
                    total_results[method][r]['true'] += results_dmv[method][r]['true']

                    if r not in avg_aux[method].keys():
                        avg_aux[method][r] = {}

                    aux = get_all_stats(results_dmv[method][r]['inferred'], results_dmv[method][r]['true'],
                                        labels=[-1, 0, 1])

                    for key in aux.keys():
                        if key not in avg_aux[method][r].keys():
                            avg_aux[method][r][key] = aux[key] / num_iterations
                        else:
                            avg_aux[method][r][key] += aux[key] / num_iterations

        for method in methods:
            for r in r_values:
                total_weighted_stats[method][r] = get_all_stats(total_results[method][r]['inferred'],
                                                                total_results[method][r]['true'], labels=[-1, 0, 1])

        print(f"Results for Hierarchical with {num_nodes} communities, {num_comm} communities, "
              f"power coefficient{gamma} and opinion generator mv")
        for method in avg_aux.keys():
            print(f"\n{method} :")
            for r in avg_aux[method].keys():
                print(f"\t{r} : ")
                for key2 in avg_aux[method][r].keys():
                    print(f"\t Iterations averaged {key2} : {avg_aux[method][r][key2]}")
                    print(f"\t Inferred nodes averaged {key2} : {total_weighted_stats[method][r][key2]}")

        #######
        # LP  #
        #######
        total_results = {method: {r: {'inferred': [], 'true': []} for r in r_values} for method in methods}
        total_weighted_stats = {method: {r: {} for r in r_values} for method in methods}
        avg_aux = {}
        for _ in range(num_iterations):
            global_deg_seq = generate_power_law_degree_sequence(num_nodes, gamma=gamma, k_min=10)
            out_deg = [0] * num_nodes
            in_deg = [0] * num_nodes
            for i, deg in enumerate(global_deg_seq):
                if (deg - deg // 10) % 2 == 0:
                    in_deg[i] = deg - deg // 10
                    out_deg[i] = deg // 10
                else:
                    in_deg[i] = deg - deg // 10 + 1
                    out_deg[i] = deg // 10 - 1
            G = generate_hierarchical_configuration_model(ext_degree_sequence=out_deg,
                                                          in_degree_sequence=in_deg,
                                                          community_sizes=comms)
            v = random.sample(list(G.nodes()), selected_nodes_per_graph)  # choose a random set of nodes
            opinion_dist = OpinionDistribution(G)  # create instance of class OpinionDistribution with graph G
            opinion_dist.initialize_opinions(states=[-1, 0, 1], probabilities=[1 / 3, 1 / 3, 1 / 3],
                                             label='opinion')
            opinion_dist.opinion_generator_discrete_label_propagation(label='opinion',
                                                                      num_iterations=num_nodes // 100)
            graph_inf = GraphInference(opinion_dist.graph)
            results_dmv = graph_inf.do_inference(node_set=v, radius_values=r_values, methods=methods,
                                                 label='opinion',
                                                 count_results=2, clear_results=False, num_iterations=1)
            proportion_of_labels(num_communities=num_comm, nodes_per_comm=num_nodes // num_comm,
                                 Graph=G, label='opinion')

            for method in methods:
                if method not in avg_aux.keys():
                    avg_aux[method] = {}

                for r in r_values:
                    total_results[method][r]['inferred'] += results_dmv[method][r]['inferred']
                    total_results[method][r]['true'] += results_dmv[method][r]['true']

                    if r not in avg_aux[method].keys():
                        avg_aux[method][r] = {}

                    aux = get_all_stats(results_dmv[method][r]['inferred'], results_dmv[method][r]['true'],
                                        labels=[-1, 0, 1])

                    for key in aux.keys():
                        if key not in avg_aux[method][r].keys():
                            avg_aux[method][r][key] = aux[key] / num_iterations
                        else:
                            avg_aux[method][r][key] += aux[key] / num_iterations

        for method in methods:
            for r in r_values:
                total_weighted_stats[method][r] = get_all_stats(total_results[method][r]['inferred'],
                                                                total_results[method][r]['true'], labels=[-1, 0, 1])

        print(f"Results for Hierarchical with {num_nodes} communities, {num_comm} communities, "
              f"power coefficient{gamma} and opinion generator lp")
        for method in avg_aux.keys():
            print(f"\n{method} :")
            for r in avg_aux[method].keys():
                print(f"\t{r} : ")
                for key2 in avg_aux[method][r].keys():
                    print(f"\t Iterations averaged {key2} : {avg_aux[method][r][key2]}")
                    print(f"\t Inferred nodes averaged {key2} : {total_weighted_stats[method][r][key2]}")

    # ########
    # # PAC  #
    # ########
    # for added_edges in added_edges_per_step:
    #     total_results = {method: {r: {'inferred': [], 'true': []} for r in r_values} for method in methods}
    #     total_weighted_stats = {method: {r: {} for r in r_values} for method in methods}
    #     avg_aux = {}
    #     for n in range(num_iterations):
    #         G = preferential_attachment_with_colors(num_nodes=num_nodes, num_edges=added_edges, labels=labels_prob,
    #                                                 size_init_graph=size_ini, label='opinion', delta=delta)
    #         v = random.sample(list(G.nodes()), selected_nodes_per_graph)  # choose a random set of nodes
    #         graph_inf = GraphInference(G)
    #         results_dmv = graph_inf.do_inference(node_set=v, radius_values=r_values, methods=methods, label='opinion',
    #                                              count_results=2, clear_results=False, num_iterations=1)
    #         proportion_of_labels_total(Graph=G, label='opinion')
    #         for method in methods:
    #             if method not in avg_aux.keys():
    #                 avg_aux[method] = {}
    #
    #             for r in r_values:
    #                 total_results[method][r]['inferred'] += results_dmv[method][r]['inferred']
    #                 total_results[method][r]['true'] += results_dmv[method][r]['true']
    #
    #                 if r not in avg_aux[method].keys():
    #                     avg_aux[method][r] = {}
    #
    #                 aux = get_all_stats(results_dmv[method][r]['inferred'], results_dmv[method][r]['true'],
    #                                     labels=[-1, 0, 1])
    #
    #                 for key in aux.keys():
    #                     if key not in avg_aux[method][r].keys():
    #                         avg_aux[method][r][key] = aux[key] / num_iterations
    #                     else:
    #                         avg_aux[method][r][key] += aux[key] / num_iterations
    #
    #     for method in methods:
    #         for r in r_values:
    #             total_weighted_stats[method][r] = get_all_stats(total_results[method][r]['inferred'],
    #                                                             total_results[method][r]['true'], labels=[-1, 0, 1])
    #
    #     print(f"Results for Pref. Attach. with Colors with {num_nodes} nodes, {added_edges} added edges per step, "
    #           f" {size_ini} initial size, and {labels_prob} labels probabilities "
    #           f"over {selected_nodes_per_graph} of the node set.")
    #     for method in avg_aux.keys():
    #         print(f"\n{method} :")
    #         for r in avg_aux[method].keys():
    #             print(f"\t{r} : ")
    #             for key2 in avg_aux[method][r].keys():
    #                 print(f"\t Iterations averaged {key2} : {avg_aux[method][r][key2]}")
    #                 print(f"\t Inferred nodes averaged {key2} : {total_weighted_stats[method][r][key2]}")

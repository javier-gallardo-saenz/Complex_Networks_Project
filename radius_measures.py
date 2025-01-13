from generate_graphs import *
from graph_inference import *

num_nodes = 10000
num_comm = 100
comms = [num_nodes//num_comm] * num_comm
prob_edge = 0.01
gamma = 2 #power law
deg_seq = generate_power_law_degree_sequence(num_nodes, gamma=gamma, k_min=10)
num_edges = 1
prob_intra = 0.4
prob_inter = 0.0002
gamma_out = 7
gamma_in = 2.5
out_deg_skewed = generate_power_law_degree_sequence(num_nodes, gamma=gamma_out, k_min=1)
out_deg = [x - 1 for x in out_deg_skewed]
in_deg = [
    degree
    for _ in range(num_comm)
    for degree in generate_power_law_degree_sequence(num_nodes//num_comm, gamma=gamma_in, k_min=10)
]
r = 2  # radius of the known ball
methods = {'dmv', 'dwmv', 'dvm', 'dlp'}
selected_nodes_per_graph = num_nodes//100


#G = generate_erdos_renyi_graph(n=num_nodes, p=prob_edge)
#G = generate_configuration_model(degree_sequence=deg_seq)
#G = generate_albert_barabasi_graph(n=num_nodes, m=num_edges)
#G = generate_sbm(sizes_sbm=comms, p_inter=prob_inter, p_intra=prob_intra)
G = generate_hierarchical_configuration_model(ext_degree_sequence=out_deg,
                                                  in_degree_sequence=in_deg,
                                                  community_sizes=comms)
S = random.sample(list(G.nodes()), selected_nodes_per_graph)  # choose a random set of nodes
graph_inf = GraphInference(G)
avg = 0
for v in S:
    ball, boundary = graph_inf.get_ball_and_boundary(v, r)
    avg += len(ball)

avg = avg/len(S)
print(avg)


import random
from collections import Counter
from tqdm import tqdm


def discrete_majority_voting(graph_inference, node, radius, label='opinion'):
    """
    Infers the discrete attribute label for the nodes in the boundary of a ball using label propagation
    :param graph_inference: inherits GraphInference class self and methods
    :param node: node
    :param radius: radius
    :param label: label of the attribute to be inferred

    """
    ball, boundary = graph_inference.get_ball_and_boundary(node, radius)
    inferred_label = graph_inference.name_inferred_label(label, node, radius, 'dmv')

    #shuffle nodes in boundary to ensure they are visited at a random order
    shuffled_boundary = list(boundary)
    random.shuffle(shuffled_boundary)

    for node in shuffled_boundary:
        # Finds neighbors in the ball
        neighbors_in_ball = set(graph_inference.graph.neighbors(node)) & ball
        neighbor_opinions = [graph_inference.graph.nodes[neighbor][label] for neighbor in neighbors_in_ball]
        opinion_counts = Counter(neighbor_opinions)
        majority_opinion = opinion_counts.most_common(1)[0][0]
        graph_inference.graph.nodes[node][inferred_label] = majority_opinion


def discrete_weighted_majority_voting(graph_inference, node, radius, label='opinion'):
    """
    Infers the discrete attribute label for the nodes in the boundary of a ball
     using weighted majority voting by the degree of the sampled nodes

    :param graph_inference: inherits GraphInference class self and methods
    :param node: node
    :param radius: radius
    :param label: label of the attribute to be inferred
    """

    ball, boundary = graph_inference.get_ball_and_boundary(node, radius)
    inferred_label = graph_inference.name_inferred_label(label, node, radius, 'dwmv')

    # shuffle nodes in boundary to ensure they are visited at a random order
    shuffled_boundary = list(boundary)
    random.shuffle(shuffled_boundary)

    for node in shuffled_boundary:
        weights = {}
        neighbors_in_ball = set(graph_inference.graph.neighbors(node)) & ball
        for neighbor in neighbors_in_ball:
            opinion = graph_inference.graph.nodes[neighbor][label]
            # Each opinion is counted as many times as the degree of the node
            weight = graph_inference.graph.degree(neighbor)
            if opinion in weights:
                weights[opinion] += weight
            else:
                weights[opinion] = weight

        #Get opinion with the most weight
        majority_opinion = max(weights.items(), key=lambda x: x[1])[0]
        graph_inference.graph.nodes[node][inferred_label] = majority_opinion


def discrete_voter_model(graph_inference, node, radius, label='opinion', num_iterations=10):
    """
    Infers the discrete attribute label for every node in the graph by assigning them the label 
    of one of their neighbors chosen randomly.
    :param graph_inference: inherits GraphInference class self and methods
    :param node: node
    :param radius: radius
    :param num_iterations: number of iterations the process will be done
    :param label: label of the attribute to be inferred
    """
    ball, boundary = graph_inference.get_ball_and_boundary(node, radius)
    inferred_label = graph_inference.name_inferred_label(label, node, radius, 'dvm')

    for _ in range(num_iterations):

        # shuffle nodes in boundary to ensure they are visited at a random order
        shuffled_boundary = list(boundary)
        random.shuffle(shuffled_boundary)

        for node in shuffled_boundary:
            neighbors_in_ball = set(graph_inference.graph.neighbors(node)) & ball
            random_neighbor = random.choice(list(neighbors_in_ball))
            graph_inference.graph.nodes[node][inferred_label] = graph_inference.graph.nodes[random_neighbor][label]




def discrete_label_propagation(graph_inference, node, radius, label='opinion', num_iterations=10):
    """
    Infers the discrete attribute label for the nodes in the boundary of a ball using label propagation.
    Each iteration sweeps through all the nodes in the boundary.

    :param graph_inference: inherits GraphInference class self and methods
    :param node: node
    :param radius: radius
    :param label: label of the attribute to be inferred
    :param num_iterations: number of steps

    """
    ball, boundary = graph_inference.get_ball_and_boundary(node, radius)
    inferred_label = graph_inference.name_inferred_label(label, node, radius, 'dlp')

    # shuffle nodes in boundary
    shuffled_boundary = list(boundary)
    random.shuffle(shuffled_boundary)

    for node in shuffled_boundary:
        # first infer an opinion for every node in boundary
        neighbors_in_ball = set(graph_inference.graph.neighbors(node)) & ball
        neighbor_opinions = [graph_inference.graph.nodes[neighbor][label] for neighbor in neighbors_in_ball]
        opinion_counts = Counter(neighbor_opinions)
        majority_opinion = opinion_counts.most_common(1)[0][0]
        graph_inference.graph.nodes[node][inferred_label] = majority_opinion

    aux_str = "Performing label propagation after all nodes in boundary have had an initial opinion assigned"
    #for _ in tqdm(range(num_iterations)):
    for _ in range(num_iterations-1):
        # now perform label propagation on them
        node = random.choice(list(boundary))
        neighbors_in_ball = set(graph_inference.graph.neighbors(node)) & ball
        neighbor_opinions = [graph_inference.graph.nodes[neighbor][label] for neighbor in neighbors_in_ball]
        opinion_counts = Counter(neighbor_opinions)
        majority_opinion = opinion_counts.most_common(1)[0][0]
        graph_inference.graph.nodes[node][inferred_label] = majority_opinion




# ----------------------------------------------------
# Old functions that are now implemented as class Graph Inference methods
# ----------------------------------------------------

# def majority_vote_inference(G, sampled_nodes, sampled_opinions, boundary):
#     """
#     Infierne opiniones nodos en la frontera externa mediante votación mayoritaria entre sus vecinos muestreados.
#
#     Parámetros:
#     - G (networkx.Graph): El grafo.
#     - sampled_nodes (set): Conjunto de IDs de nodos muestreados.
#     - sampled_opinions (dict): Diccionario de nodos muestreados y sus opiniones.
#     - boundary (set): Conjunto de IDs de nodos en la frontera externa.
#
#     Retorna:
#     - inferred_opinions (dict): Diccionario de opiniones inferidas para nodos en la frontera externa.
#     """
#     inferred_opinions = {}
#     for node in boundary:
#         # Encuentra vecinos del nodo que están en el conjunto muestreados
#         neighbors_in_S = set(G.neighbors(node)) & sampled_nodes
#         if neighbors_in_S:
#             neighbor_opinions = [sampled_opinions[neighbor] for neighbor in neighbors_in_S]
#             opinion_counts = Counter(neighbor_opinions)
#             majority_opinion = opinion_counts.most_common(1)[0][0]
#             inferred_opinions[node] = majority_opinion
#         else:
#             inferred_opinions[node] = 0  # Por defecto, votante indeciso si no hay vecinos muestreados
#     return inferred_opinions
#
#
# def weighted_majority_vote_inference(G, sampled_nodes, sampled_opinions, boundary):
#     """
#     Infierne las opiniones utilizando votación mayoritaria ponderada por el grado de los nodos muestreados.
#
#     Parámetros:
#     - G (networkx.Graph): El grafo.
#     - sampled_nodes (set): Nodos muestreados.
#     - sampled_opinions (dict): Opiniones de los nodos muestreados.
#     - boundary (set): Nodos en la frontera.
#
#     Retorna:
#     - inferred_opinions (dict): Opiniones inferidas para los nodos en la frontera.
#     """
#     inferred_opinions = {}
#     degrees = dict(G.degree())
#     for node in boundary:
#         neighbors_in_S = set(G.neighbors(node)) & sampled_nodes
#         if neighbors_in_S:
#             weights = {}
#             for neighbor in neighbors_in_S:
#                 opinion = sampled_opinions[neighbor]
#                 weight = degrees[neighbor]  # Ponderación por grado
#                 if opinion in weights:
#                     weights[opinion] += weight
#                 else:
#                     weights[opinion] = weight
#             # Obtener la opinión con mayor peso
#             majority_opinion = max(weights.items(), key=lambda x: x[1])[0]
#             inferred_opinions[node] = majority_opinion
#         else:
#             inferred_opinions[node] = 0  # Valor por defecto
#     return inferred_opinions
#
#
# def label_propagation_inference(G, sampled_nodes, sampled_opinions, boundary, max_iter=1000):
#     """
#     Infierne las opiniones utilizando el algoritmo de Label Propagation (original).
#
#     Parámetros:
#     - G (networkx.Graph): El grafo.
#     - sampled_nodes (set): Nodos muestreados.
#     - sampled_opinions (dict): Opiniones de los nodos muestreados.
#     - boundary (set): Nodos en la frontera.
#     - max_iter (int): Número máximo de iteraciones.
#
#     Retorna:
#     - inferred_opinions (dict): Opiniones inferidas para los nodos en la frontera.
#     """
#     # Preparar datos para LabelPropagation
#     nodes = list(G.nodes())
#     node_index = {node: idx for idx, node in enumerate(nodes)}
#     num_nodes = len(nodes)
#
#     # Crear matriz de adyacencia
#     adjacency = nx.to_scipy_sparse_array(G, format='csr')
#
#     # Map labels to 0,1,2
#     label_mapping = {-1: 0, 0: 1, 1: 2}
#     inverse_label_mapping = {v: k for k, v in label_mapping.items()}
#
#     labels = np.full(num_nodes, -1)
#     for node in sampled_nodes:
#         labels[node_index[node]] = label_mapping[sampled_opinions[node]]
#
#     # Aplicar LabelPropagation
#     label_prop = LabelPropagation(max_iter=max_iter)
#     label_prop.fit(adjacency, labels)
#
#     # Extraer las etiquetas inferidas para la frontera
#     inferred_opinions = {}
#     for node in boundary:
#         inferred_label = label_prop.transduction_[node_index[node]]
#         inferred_opinions[node] = inverse_label_mapping.get(int(inferred_label), 0)
#     return inferred_opinions
#
#
# def voter_model_inference(G, sampled_nodes, sampled_opinions, boundary, num_simulations=10, num_steps=5000):
#     """
#     Infierne las opiniones re-ejecutando el modelo del votante múltiples veces y tomando la opinión más frecuente.
#
#     Parámetros:
#     - G (networkx.Graph): El grafo.
#     - sampled_nodes (set): Nodos muestreados.
#     - sampled_opinions (dict): Opiniones de los nodos muestreados.
#     - boundary (set): Nodos en la frontera.
#     - num_simulations (int): Número de simulaciones a ejecutar.
#     - num_steps (int): Número de pasos en cada simulación.
#
#     Retorna:
#     - inferred_opinions (dict): Opiniones inferidas para los nodos en la frontera.
#     """
#     opinions_counts = {node: Counter() for node in boundary}
#
#     for _ in tqdm(range(num_simulations), desc="Running Voter Model Simulations"):
#         # Inicializar opiniones
#         opinions = {}
#         for node in G.nodes():
#             if node in sampled_nodes:
#                 opinions[node] = sampled_opinions[node]
#             else:
#                 opinions[node] = np.random.choice([-1, 0, 1])
#
#         # Simular modelo del votante
#         opinions = simulate_voter_model(G, opinions, num_steps=num_steps, stubborn_fraction=0)
#
#         # Registrar opiniones de los nodos en la frontera
#         for node in boundary:
#             opinions_counts[node][opinions[node]] += 1
#
#     # Determinar la opinión más frecuente para cada nodo en la frontera
#     inferred_opinions = {}
#     for node in boundary:
#         majority_opinion = opinions_counts[node].most_common(1)[0][0]
#         inferred_opinions[node] = majority_opinion
#
#     return inferred_opinions
#
#
# def bayesian_inference(G, sampled_nodes, sampled_opinions, boundary, beta=1.0, max_iter=50, tol=1e-5):
#     """
#     Infers opinions using belief propagation (Bayesian inference).
#
#     Parameters:
#     - G (networkx.Graph): The graph.
#     - sampled_nodes (set): Set of sampled node IDs.
#     - sampled_opinions (dict): Dictionary of sampled nodes and their opinions.
#     - boundary (set): Set of boundary node IDs.
#     - beta (float): Interaction strength parameter.
#     - max_iter (int): Maximum number of iterations.
#     - tol (float): Convergence tolerance.
#
#     Returns:
#     - inferred_opinions (dict): Dictionary of inferred opinions for boundary nodes.
#     """
#     # Possible opinions
#     possible_opinions = [-1, 0, 1]
#     num_states = len(possible_opinions)
#
#     # Mapping from opinion to index
#     opinion_to_index = {opinion: idx for idx, opinion in enumerate(possible_opinions)}  # diccionario {-1:0, 0:1, 1:2}
#     index_to_opinion = {idx: opinion for idx, opinion in enumerate(possible_opinions)}  # diccionario {0:-1, 1:0, 2:1}
#
#     # Initialize messages: For each directed edge, store a message (array of size num_states)
#     messages = {}
#     for edge in G.edges():
#         u, v = edge
#         messages[(u, v)] = np.ones(
#             num_states) / num_states  # asigna vector (1/3, 1/3, 1/3) a cada edge (porque hay 3 stages)
#         messages[(v, u)] = np.ones(num_states) / num_states
#
#     # For observed nodes, fix their messages
#     observed_messages = {}
#     for node in sampled_nodes:
#         observed_opinion = sampled_opinions[node]
#         fixed_message = np.zeros(num_states)
#         fixed_message[opinion_to_index[observed_opinion]] = 1.0
#         for neighbor in G.neighbors(node):
#             messages[(node, neighbor)] = fixed_message.copy()
#         observed_messages[node] = fixed_message.copy()
#
#     # Iterative message passing
#     for iteration in range(max_iter):
#         delta = 0  # Change in messages for convergence check
#         new_messages = {}
#         for edge in G.edges():
#             for direction in [(edge[0], edge[1]), (edge[1], edge[0])]:
#                 i, j = direction
#                 if i in sampled_nodes:
#                     continue  # Messages from observed nodes are fixed
#                 # Compute the new message from i to j
#                 product = np.ones(num_states)
#                 for k in G.neighbors(i):
#                     if k != j:
#                         product *= messages[(k, i)]
#                 # Multiply by node potential (uniform for unobserved nodes)
#                 # Compute the message
#                 m_ij = np.zeros(num_states)
#                 for xi in possible_opinions:
#                     idx_i = opinion_to_index[xi]
#                     sum_over_xj = 0
#                     for xj in possible_opinions:
#                         idx_j = opinion_to_index[xj]
#                         # Edge potential: favor same opinions
#                         if xi == xj:
#                             edge_potential = np.exp(beta)
#                         else:
#                             edge_potential = np.exp(-beta)
#                         sum_over_xj += edge_potential * product[idx_i]
#                     m_ij[idx_i] = sum_over_xj
#                 # Normalize message
#                 m_ij /= np.sum(m_ij)
#                 # Update delta
#                 delta += np.sum(np.abs(m_ij - messages[(i, j)]))
#                 new_messages[(i, j)] = m_ij
#         # Update messages
#         for key in new_messages:
#             messages[key] = new_messages[key]
#         # Check convergence
#         if delta < tol:
#             print(f"Belief propagation converged after {iteration + 1} iterations.")
#             break
#     else:
#         print(f"Belief propagation did not converge after {max_iter} iterations.")
#
#     # Compute marginals
#     marginals = {}
#     for node in G.nodes():
#         if node in sampled_nodes:
#             marginals[node] = observed_messages[node]
#         else:
#             # Compute the product of incoming messages
#             incoming_messages = []
#             for neighbor in G.neighbors(node):
#                 incoming_messages.append(messages[(neighbor, node)])
#             marginal = np.ones(num_states)
#             for msg in incoming_messages:
#                 marginal *= msg
#             # Normalize
#             marginal /= np.sum(marginal)
#             marginals[node] = marginal
#
#     # Infer opinions for boundary nodes
#     inferred_opinions = {}
#     for node in boundary:
#         marginal = marginals[node]
#         inferred_state_index = np.argmax(marginal)
#         inferred_opinion = index_to_opinion[inferred_state_index]
#         inferred_opinions[node] = inferred_opinion
#
#     return inferred_opinions
#
#
# def heuristic_based_inference(G, sampled_nodes, sampled_opinions, boundary):
#     """
#     Infers opinions using a heuristic-based method that considers node centrality.
#
#     Parameters:
#     - G (networkx.Graph): The graph.
#     - sampled_nodes (set): Set of sampled node IDs.
#     - sampled_opinions (dict): Dictionary of sampled nodes and their opinions.
#     - boundary (set): Set of boundary node IDs.
#
#     Returns:
#     - inferred_opinions (dict): Dictionary of inferred opinions for boundary nodes.
#     """
#     inferred_opinions = {}
#
#     # Compute PageRank as a centrality measure
#     pagerank = nx.pagerank(G, alpha=0.85)
#
#     for node in boundary:
#         neighbors_in_S = set(G.neighbors(node)) & sampled_nodes
#         if neighbors_in_S:
#             # Aggregate opinions weighted by PageRank
#             opinion_scores = {}
#             for neighbor in neighbors_in_S:
#                 opinion = sampled_opinions[neighbor]
#                 weight = pagerank[neighbor]
#                 if opinion in opinion_scores:
#                     opinion_scores[opinion] += weight
#                 else:
#                     opinion_scores[opinion] = weight
#             # Assign the opinion with the highest weighted score
#             majority_opinion = max(opinion_scores.items(), key=lambda x: x[1])[0]
#             inferred_opinions[node] = majority_opinion
#         else:
#             # Default to swing (0) if no sampled neighbors
#             inferred_opinions[node] = 0
#     return inferred_opinions

# def discrete_voter_model(graph_inference, num_iterations, label='opinion'):
#     """
#     Infers the discrete attribute label for every node in the graph by assigning them the label
#     of one of their neighbors chosen randomly.
#     :param graph_inference: inherits GraphInference class self and methods
#     :param num_iterations: number of iterations the process will be done
#     :param label: label of the attribute to be inferred
#     """
#     # graph_inference.initialize_opinions(label='opinion', states=[-1,1], probabilities=[0.4,0.6], opinion_values=None)
#     for _ in range(num_iterations):
#         for node in graph_inference.graph.nodes:
#             neighbors = set(graph_inference.graph.neighbors(node))
#             random_neighbor = random.choice(list(neighbors))
#             graph_inference.graph.nodes[node][label] = graph_inference.graph.nodes[random_neighbor][label]
#     return graph_inference
#
#
# def discrete_modified_biased_voter_model(graph_inference, num_iterations, delta, label='opinion'):
#     """
#     Infers the discrete attribute label for every node in the graph by assigning them the label
#     of one of their neighbors chosen randomly with probability based on 'how close' opinions are.
#     :param graph_inference: inherits GraphInference class self and methods
#     :param num_iterations: number of iterations the process will be done
#     :param delta: fixed variable >=0 to fix a minimum probability for every vertex
#                   to change their label to their neighbor's
#     :param label: label of the attribute to be inferred
#     """
#     for _ in range(num_iterations):
#         for node in graph_inference.graph.nodes:
#             neighbors = set(graph_inference.graph.neighbors(node))
#             random_neighbor = random.choice(list(neighbors))
#             prob = abs(graph_inference.graph.nodes[node][label] + graph_inference.graph.nodes[random_neighbor][label]+delta)/(2+delta)
#             if random() < prob:
#                 graph_inference.graph.nodes[node][label] = graph_inference.graph.nodes[random_neighbor][label]
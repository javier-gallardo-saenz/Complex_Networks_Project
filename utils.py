import networkx as nx


# ----------------------------------------------------
# Graph loading utils (NO LONGER CLASS GRAPH_INFERENCE)
# ----------------------------------------------------

def load_snap_graph(file_path, directed=False):
    """
    Carga un grafo desde un archivo de SNAP.

    Parámetros:
    - file_path (str): Ruta al archivo de SNAP (edge list).
    - directed (bool): Si True, crea un grafo dirigido. Por defecto, False.

    Retorna:
    - G (networkx.Graph): Un grafo cargado desde el archivo de SNAP.
    """
    print(f"Cargando grafo de SNAP desde {file_path}...")
    G = nx.read_edgelist(file_path, create_using=nx.DiGraph() if directed else nx.Graph(), nodetype=int)

    # Preprocesamiento
    if not nx.is_connected(G.to_undirected()):
        print("El grafo de SNAP no está conectado. Extrayendo el componente conectado más grande.")
        G = G.subgraph(max(nx.connected_components(G.to_undirected()), key=len)).copy()
        print(f"Grafo reducido al componente conectado más grande: {G.number_of_nodes()} nodos.")

    # Eliminar bucles
    G.remove_edges_from(nx.selfloop_edges(G))
    print("Grafo de SNAP cargado y preprocesado correctamente.")

    # Si el grafo es dirigido y tus funciones esperan grafos no dirigidos, convertir a no dirigido
    if directed:
        G = G.to_undirected()
        print("Convirtiendo grafo dirigido a no dirigido para compatibilidad con las funciones existentes.")

    return G

def proportion_of_labels(num_communities, nodes_per_comm, Graph, label):
    if len(Graph.nodes) != num_communities*nodes_per_comm:
        raise ValueError("Number of nodes of the graph must be equal to the number of communities times the number of nodes per community.")
    labels = {}
    for n in range(num_communities):
        labels[n] = {}
        for node in range(n*nodes_per_comm, (n+1)*nodes_per_comm):
            if Graph.nodes[node][label] is None:
                raise KeyError(f"The node {node} does not have the label '{label}'.")
            if Graph.nodes[node][label] not in labels[n].keys():
                labels[n][Graph.nodes[node][label]] = 1
            else:
                labels[n][Graph.nodes[node][label]] += 1
        print(f"In community {n} the proportion of {[key for key in labels[n].keys()]} \'s is {[prop/nodes_per_comm for prop in labels[n].values()]}")

def proportion_of_labels_total(Graph, label):
    labels = {}
    for node in list(Graph.nodes):
        if Graph.nodes[node][label] is None:
            raise KeyError(f"The node {node} does not have the label '{label}'.")
        if Graph.nodes[node][label] not in labels.keys():
            labels[Graph.nodes[node][label]] = 1
        else:
            labels[Graph.nodes[node][label]] += 1
    print(f"The proportion of {[key for key in labels.keys()]} \'s is {[prop/len(Graph.nodes) for prop in labels.values()]}")






















# ----------------------------------------------------
# Old functions that are now implemented in a class
# ----------------------------------------------------

# def ball_of_radius(graph, v, r):
#     """
#     Computes the ball of radius r centered on a node v in a graph.
#
#     Parameters:
#     - graph: networkx.Graph, the input graph
#     - v: the vertex from which distances are measured
#     - r: the radius (non-negative integer)
#
#     Returns:
#     - A set of nodes at distance <= r from node v
#     """
#     # Use NetworkX's single-source shortest paths function
#     lengths = nx.single_source_shortest_path_length(graph, v, cutoff=r)
#     # Return all nodes with distance <= r
#     return set(lengths.keys())
#
#
# def boundary_of_ball(G, v=0, r=0, nodes_in_ball=None):
#     """
#     Computes the boundary of a ball centered on a node v in a graph.
#
#     Parameters:
#         - G: networkx.Graph, the input graph
#         - v: the vertex from which distances are measured
#         - r: the radius (non-negative integer)
#         - nodes_in_ball: precalculated set of nodes in ball centered on node v
#
#     Returns:
#         - Frontier of ball of radius r centered in node v
#     """
#
#     #if a ball is not provided, the ball of radius r centered in v is calculated
#     if nodes_in_ball is None:
#         nodes_in_ball = ball_of_radius(G, v, r)
#
#     boundary = set()
#     for node in nodes_in_ball:
#         neighbors = set(G.neighbors(node))
#         boundary.update(neighbors - nodes_in_ball)
#
#     return boundary

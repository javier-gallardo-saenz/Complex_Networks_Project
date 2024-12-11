import networkx as nx

def ball_of_radius(graph, v, r):
    """
    Computes the ball of radius r centered on a node v in a graph.

    Parameters:
    - graph: networkx.Graph, the input graph
    - v: the vertex from which distances are measured
    - r: the radius (non-negative integer)

    Returns:
    - A set of nodes at distance <= r from node v
    """
    # Use NetworkX's single-source shortest paths function
    lengths = nx.single_source_shortest_path_length(graph, v, cutoff=r)
    # Return all nodes with distance <= r
    return set(lengths.keys())

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

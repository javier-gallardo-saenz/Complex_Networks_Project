import networkx as nx
from inference_models import *


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



# ----------------------------------------------------
# THE CLASS THAT DOES EVERYTHING!!!!
# ----------------------------------------------------
class GraphInference:
    def __init__(self, graph):
        """
        Initialize with a graph object.
        """
        self.graph = graph
        self.cache = {}  # To store precomputed balls and frontiers
        self.inference_cache = {}  # To store already computed inference methods over each key (node, radius)


    # ----------------------------------------------------
    # Get ball, get frontier of ball
    # ----------------------------------------------------
    def ball_of_radius(self, v, r):
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
        lengths = nx.single_source_shortest_path_length(self.graph, v, cutoff=r)
        # Return all nodes with distance <= r
        return set(lengths.keys())

    def ball_and_boundary(self, node, radius):
        """
        Computes the ball of radius r centered on a node v in a graph and its boundary

        parameters:
        - node: the node from which distances are measured
        - radius: the radius (non-negative integer)

        returns:
        - A set of nodes at distance <= r from node v
        - The boundary of that set
        """
        nodes_in_ball = self.ball_of_radius(node, radius)
        boundary = set()
        for node in nodes_in_ball:
            neighbors = set(self.graph.neighbors(node))
            boundary.update(neighbors - nodes_in_ball)

        return nodes_in_ball, boundary

    def get_ball_and_boundary(self, node, radius):
        """
        Retrieve the cached ball and frontier or compute them if not already cached.
        """
        key = (node, radius)
        if key not in self.cache:
            self.cache[key] = self.ball_and_boundary(node, radius)
        return self.cache[key]

    # ----------------------------------------------------
    # Utilities
    # ----------------------------------------------------
    def clear_cache(self):
        self.cache = {}


    @staticmethod
    def name_inferred_label(label, node, radius, method_name):
        """
        Standardized labeling of an inferred attribute on the boundary of the ball of radius centered in node

        :param label: label over which inference is going to be performed
        :param node: node that defines the "known" ball
        :param radius: radius of the "known" ball
        :param method_name: (string) name of the inference method used
        """
        return label + '_inferred_' + str(node) + '_' + str(radius) + '_' + method_name

    # ----------------------------------------------------
    # Inference methods
    # ----------------------------------------------------
    def discrete_majority_voting(self, node, radius, label='opinion'):
        discrete_majority_voting(self, node, radius, label)





















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

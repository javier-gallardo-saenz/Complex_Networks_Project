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


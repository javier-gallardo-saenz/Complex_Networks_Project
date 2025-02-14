import networkx as nx

def visualize_graph(G, labels, edge_color='lightgrey'):
    """
    Visualizes the graph with nodes colored by their labels.
    
    Parameters:
    - G (networkx.Graph): The input graph.
    - labels (dict): A dictionary where keys are node IDs and values are the node labels.
    """
    import matplotlib.pyplot as plt
    
    # Get the list of unique labels
    unique_labels = list(set(labels.values()))
    sorted_labels = sorted(unique_labels)
    
    # Create a color map based on labels
    color_map = {label: plt.cm.jet(i / len(sorted_labels)) for i, label in enumerate(sorted_labels)}
    node_colors = [color_map[labels[node]] for node in G.nodes()]
    
    # Draw the graph
    plt.figure(figsize=(10, 8))
    nx.draw(G, node_color=node_colors, with_labels=True, node_size=50, font_size=10, edge_color=edge_color)
   # plt.title("Label Propagation - Community Detection")
    plt.show()
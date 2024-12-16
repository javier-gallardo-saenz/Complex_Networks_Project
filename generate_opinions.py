import networkx as nx
import numpy as np
from tqdm import tqdm
import random


def all_nodes_labeled(graph, attribute='label'):
    """
    Check if all nodes in the graph have a specific attribute.

    Args:
        graph: A NetworkX graph object.
        attribute: The name of the attribute to check (default is 'label').

    Returns:
        True if all nodes have the specified attribute, False otherwise.
    """
    return all(attribute in graph.nodes[node] for node in graph.nodes)


class OpinionDistribution:
    def __init__(self, graph=None):
        """
        Initialize the OpinionDistribution class.

        Args:
            graph (optional): An existing graph object. If provided, it will be used.
        """

        self.graph = graph

    def initialize_opinions(self, label='opinion', states=None, probabilities=None, opinion_values=None):
        """
        Assigns initial opinion values to all nodes in the graph.

        :param label: The name of the attribute that is about to be generated
        :param states: A list with the possible states of the 'label' attribute
        :param probabilities: A list with the probabilities associated to each state of the 'label' attribute
        :param opinion_values: An optional dictionary with predetermined values for the 'label' attribute
        """

        if opinion_values is not None:
            if isinstance(opinion_values, dict):
                # Use the provided dictionary
                nx.set_node_attributes(self.graph, name=label, values=opinion_values)
            else:
                raise ValueError("opinion_values must be a dictionary")
        else:
            if probabilities is None:
                probabilities = [0.4, 0.2, 0.4]
            if states is None:
                states = [-1, 0, 1]

            for node in self.graph.nodes():
                self.graph.nodes[node][label] = np.random.choice(states, p=probabilities)


    def get_opinion(self, node, label='opinion'):
        """
        Retrieve the attribute of a specific node.

        Args:
            node: The node for which to retrieve the opinion.
            label (optional): The name of the attribute to be retrieved
        """
        if label not in self.graph.nodes[node]:
            raise ValueError('The opinion attribute does not exist for the given node.')

        return self.graph.nodes[node][label]

    def basic_opinion_generator(self, label='opinion', num_steps=10000, emergency_states=None, emergency_probs=None):
        """
        :param label: attribute to update.
        :param num_steps: (Integer) Number of time-steps to run the generating model for
        :param emergency_states: (List) List of emergency states to use.
        :param emergency_probs: (List) List of emergency probabilities to use.

        :return: Opinion list for the graph
        """
        if emergency_probs is None:
            emergency_probs = [0.4, 0.2, 0.4]
        if emergency_states is None:
            emergency_states = [-1, 0, 1]

        for _ in tqdm(range(num_steps), desc="Evolving initial attribute distribution"):
            node = random.choice(list(self.graph.nodes()))
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                neighbor = random.choice(neighbors)
                if label in self.graph.nodes[neighbor]:
                    self.graph.nodes[node][label] = self.graph.nodes[neighbor][label]
                else:
                    #pathological cases when there are nodes that have not been assigned an initial label value
                    if label in self.graph.nodes[node]:
                        self.graph.nodes[neighbor][label] = self.graph.nodes[node][label]
                    else:
                        self.graph.nodes[node][label] = np.random.choice(emergency_states, p=emergency_probs)
                        self.graph.nodes[neighbor][label] = np.random.choice(emergency_states, p=emergency_probs)







# ----------------------------------------------------
# Old functions that are now implemented in a class
# ----------------------------------------------------

def assign_initial_opinions(G, states=None, probabilities=None):
    """
    Asigna opiniones iniciales a cada nodo del grafo basándose en probabilidades especificadas.

    Parámetros:
    - G (networkx.Graph): El grafo.
    - states (list): Estados de opinión posibles (por defecto: -1, 0, 1).
    - probabilities (list): Probabilidades correspondientes para cada estado.

    Retorna:
    - opinions (dict): Diccionario que mapea cada nodo a su opinión inicial.
    """
    if probabilities is None:
        probabilities = [0.4, 0.2, 0.4]
    if states is None:
        states = [-1, 0, 1]
    opinions = {}
    for node in G.nodes():
        opinions[node] = np.random.choice(states, p=probabilities)
    return opinions


def basic_opinion_generator(G, states=None, probabilities=None, num_steps=10000):
    """

    :param G: (Networkx graph) Graph
    :param states: (List) Possible opinions
    :param probabilities: (List) Probability of each opinion appearing in a node at time t=0
    :param num_steps: (Integer) Number of time-steps to run the generating model for
    :return: Opinion list for the graph
    """
    if probabilities is None:
        probabilities = [0.4, 0.2, 0.4]
    if states is None:
        states = [-1, 0, 1]
    opinions = assign_initial_opinions(G, states, probabilities)

    for _ in tqdm(range(num_steps), desc="Generating initial opinion distribution"):
        node = random.choice(list(G.nodes()))
        neighbors = list(G.neighbors(node))
        if neighbors:
            neighbor = random.choice(neighbors)
            opinions[node] = opinions[neighbor]

    return opinions

def opinion_generator_biased_voter_model(G, node, label='opinion', num_iterations=1000,
                                         delta=0.1):
    """
    Infers the discrete attribute label for every node in the graph by assigning them the label 
    of one of their neighbors chosen randomly with probability based on 'how close' opinions are.
    :param graph_inference: inherits GraphInference class self and methods
    :param node: node
    :param radius: radius
    :param num_iterations: number of iterations the process will be done
    :param delta: fixed variable >=0 to fix a minimum probability for every vertex
                  to change their label to their neighbor's
    :param label: label of the attribute to be inferred
    """

    for _ in range(num_iterations):
        node = random.choice(list(G.nodes()))
        neighbors = set(G.neighbors(node))
        random_neighbor = random.choice(list(neighbors))
        prob = abs(G.nodes[node][label] +
                    G.nodes[random_neighbor][label]+delta)/(2+delta)
        if prob > random.random():
            G.nodes[node][label] = G.nodes[random_neighbor][label]
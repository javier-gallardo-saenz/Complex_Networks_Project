from utils import *
from inference_models import *

method_abreviations_dictionary = {
    "key1": "value1",
    "key2": "value2",
    "key3": "value3",
}


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
        self.methods = ['dmv', 'dwmv', 'dvm', 'dmbvm']

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

    def clear_inferred_opinions(self, label, node, radius, method_name):

        label_to_be_cleared = self.name_inferred_label(label, node, radius, method_name)
        _, boundary = self.get_ball_and_boundary(node, radius)

        for node in boundary:
            if label_to_be_cleared in self.graph[node]:
                del self.graph[node][label_to_be_cleared]


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
        """
        This method saves its results on the label name_inferred_label(label, node, radius, 'dmv')
        See documentation inference_models.py
        """
        discrete_majority_voting(self, node, radius, label)

    def weighted_majority_voting(self, node, radius, label='opinion'):
        """
        This method saves its results on the label name_inferred_label(label, node, radius, 'dwmv')
        See documentation inference_models.py
        """
        weighted_majority_voting(self, node, radius, label)

    def discrete_voter_model(self, node, radius, num_iterations=1000, label='opinion'):
        """
        This method saves its results on the label name_inferred_label(label, node, radius, 'dvm')
        See documentation inference_models.py
        """
        discrete_voter_model(self, node, radius, num_iterations, label)

    def discrete_modified_biased_voter_model(self, node, radius, num_iterations=1000, delta=1,
                                             label='opinion'):
        """
        This method saves its results on the label name_inferred_label(label, node, radius, 'dmbvm')
        See documentation inference_models.py
        """
        discrete_modified_biased_voter_model(self, node, radius, num_iterations, delta, label)

    def discrete_label_propagation(self, node, radius, label='opinion', num_iterations=100000):
        """
        This method saves its results on the label name_inferred_label(label, node, radius, 'dlp')
        See documentation inference_models.py
        """
        discrete_label_propagation(self, node, radius, label, num_iterations)

from utils import *
from inference_models import *

method_abbreviations_dictionary = {
    "dmv": "Discrete Majority Voting",
    "dwmv": "Discrete Weighted Majority Voting",
    "dvm": "Discrete Voter Model",
    "dmbvm": "Discrete Modified Biased Voter Model",
    "dlp": "Discrete Label Propagation"
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
        self.methods = method_abbreviations_dictionary
        # ideally this could be implemented so the names of the methods are retrieved automatically and stored here

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

    @staticmethod
    def to_set(variable):
        if isinstance(variable, int):  # If `v` is an integer
            return {variable}  # Create a set containing the integer
        elif isinstance(variable, (list, set, tuple)):  # If `v` is a list, set, or tuple
            return set(variable)  # Convert it to a set
        else:
            raise ValueError("Input must be an integer or an iterable of integers")

    def clear_cache(self):
        self.cache = {}

    def clear_inferred_opinions(self, node, radius, method_name, label='opinion'):
        """
        Clears the inferred opinions over the boundary of a given ball using a given inference method.
        """
        label_to_be_cleared = self.name_inferred_label(label, node, radius, method_name)
        _, boundary = self.get_ball_and_boundary(node, radius)

        for node in boundary:
            if label_to_be_cleared in self.graph[node]:
                del self.graph[node][label_to_be_cleared]

    def which_inference_methods(self):
        """
        Shows the available inference methods
        """
        print(self.methods)

    def get_inferred_label(self, node, radius, method_name, label='opinion'):
        """
        Gets the inferred label of a given ball using a given inference method.
        :param label: label over which inference was performed
        :param node: node that defines the "known" ball
        :param radius: radius of the "known" ball
        :param method_name: (string) name of the inference method used

        Returns a dictionary with the inferred label for each node in the boundary of the set
        """
        if method_name not in self.methods.keys():
            raise ValueError("Method '" + method_name + "' has not been implemented.")

        label_to_be_retrieved = self.name_inferred_label(label, node, radius, method_name)
        _, boundary = self.get_ball_and_boundary(node, radius)

        opinions = {}
        for node in boundary:
            if label_to_be_retrieved not in self.graph.nodes[node]:
                raise ValueError("Method '" + method_name + "' has not been executed on the given ball.")
            opinions[node] = self.graph.nodes[node][label_to_be_retrieved]

        return opinions

    def get_true_label(self, node, radius, label='opinion'):
        """
        Gets the true label of the nodes in the boundary of a ball.
        """
        _, boundary = self.get_ball_and_boundary(node, radius)

        opinions = {}
        for node in boundary:
            if label not in self.graph.nodes[node]:
                raise ValueError("Method '" + label + "' has not been initiated properly")
            opinions[node] = self.graph.nodes[node][label]

        return opinions

    # ----------------------------------------------------
    # Inference methods
    # ----------------------------------------------------

    def discrete_majority_voting(self, node_set, radius_values, label='opinion',
                                 count_results=True, clear_results=False):
        """
        This method saves its results on the label name_inferred_label(label, node, radius, 'dmv')
        See documentation inference_models.py
        """
        # Avoid double counting
        node_set = self.to_set(node_set)
        radius_values = self.to_set(radius_values)

        # first element of list results counts prediction successes
        # second element of list results keeps sum of  distance from predictions to true value
        # third element of list results counts total nodes in which we have performed inference
        results = [0] * 3

        #perform inference
        for r in radius_values:
            if not isinstance(r, int) or r < 0:  # check r is a positive integer
                raise ValueError("Radius must be a positive integer")

            for v in node_set:
                if v not in self.graph.nodes:
                    raise ValueError("Node '" + v + "' does not exist in the graph")

                discrete_majority_voting(self, v, r, label)
                if count_results:
                    inferred_label = self.get_inferred_label(v, r, 'dmv',  label)
                    true_label = self.get_true_label(v, r, label)
                    for node in inferred_label:
                        results[2] += 1
                        aux = abs(inferred_label[node] - true_label[node])
                        results[1] += aux
                        if aux == 0:
                            results[0] += 1

                self.clear_cache()
                if clear_results:
                    self.clear_inferred_opinions(v, r, label)

        if count_results:
            return results


    def discrete_weighted_majority_voting(self, node_set, radius_values, label='opinion',
                                          count_results=True, clear_results=False):
        """
        This method saves its results on the label name_inferred_label(label, node, radius, 'dwmv')
        See documentation inference_models.py
        """
        # Avoid double counting
        node_set = self.to_set(node_set)
        radius_values = self.to_set(radius_values)

        # first element of list results counts prediction successes
        # second element of list results keeps sum of  distance from predictions to true value
        # third element of list results counts total nodes in which we have performed inference
        results = [0] * 3

        # perform inference
        for r in radius_values:
            if not isinstance(r, int) or r < 0:  # check r is a positive integer
                raise ValueError("Radius must be a positive integer")

            for v in node_set:
                if v not in self.graph.nodes:
                    raise ValueError("Node '" + v + "' does not exist in the graph")

                discrete_weighted_majority_voting(self, v, r, label)
                if count_results:
                    inferred_label = self.get_inferred_label(v, r, 'dwmv', label)
                    true_label = self.get_true_label(v, r, label)
                    for node in inferred_label:
                        results[2] += 1
                        aux = abs(inferred_label[node] - true_label[node])
                        results[1] += aux
                        if aux == 0:
                            results[0] += 1

                self.clear_cache()
                if clear_results:
                    self.clear_inferred_opinions(v, r, label)

        if count_results:
            return results

    def discrete_voter_model(self, node_set, radius_values, num_iterations=1000, label='opinion',
                             count_results=True, clear_results=False):
        """
        This method saves its results on the label name_inferred_label(label, node, radius, 'dvm')
        See documentation inference_models.py
        """
        # Avoid double counting
        node_set = self.to_set(node_set)
        radius_values = self.to_set(radius_values)

        # first element of list results counts prediction successes
        # second element of list results keeps sum of  distance from predictions to true value
        # third element of list results counts total nodes in which we have performed inference
        results = [0] * 3

        # perform inference
        for r in radius_values:
            if not isinstance(r, int) or r < 0:  # check r is a positive integer
                raise ValueError("Radius must be a positive integer")

            for v in node_set:
                if v not in self.graph.nodes:
                    raise ValueError("Node '" + v + "' does not exist in the graph")

                discrete_voter_model(self, v, r, num_iterations, label)
                if count_results:
                    inferred_label = self.get_inferred_label(v, r, 'dvm', label)
                    true_label = self.get_true_label(v, r, label)
                    for node in inferred_label:
                        results[2] += 1
                        aux = abs(inferred_label[node] - true_label[node])
                        results[1] += aux
                        if aux == 0:
                            results[0] += 1

                self.clear_cache()
                if clear_results:
                    self.clear_inferred_opinions(v, r, label)

        if count_results:
            return results

    def discrete_modified_biased_voter_model(self, node_set, radius_values, num_iterations=1000, delta=1,
                                             label='opinion', count_results=True, clear_results=False):
        """
        This method saves its results on the label name_inferred_label(label, node, radius, 'dmbvm')
        See documentation inference_models.py
        """
        # Avoid double counting
        node_set = self.to_set(node_set)
        radius_values = self.to_set(radius_values)

        # first element of list results counts prediction successes
        # second element of list results keeps sum of  distance from predictions to true value
        # third element of list results counts total nodes in which we have performed inference
        results = [0] * 3

        # perform inference
        for r in radius_values:
            if not isinstance(r, int) or r < 0:  # check r is a positive integer
                raise ValueError("Radius must be a positive integer")

            for v in node_set:
                if v not in self.graph.nodes:
                    raise ValueError("Node '" + v + "' does not exist in the graph")

                discrete_modified_biased_voter_model(self, v, r, num_iterations, delta, label)
                if count_results:
                    inferred_label = self.get_inferred_label(v, r, 'dmbvm', label)
                    true_label = self.get_true_label(v, r, label)
                    for node in inferred_label:
                        results[2] += 1
                        aux = abs(inferred_label[node] - true_label[node])
                        results[1] += aux
                        if aux == 0:
                            results[0] += 1

                self.clear_cache()
                if clear_results:
                    self.clear_inferred_opinions(v, r, label)

        if count_results:
            return results

    def discrete_label_propagation(self, node_set, radius_values, label='opinion', num_iterations=100000,
                                   count_results=True, clear_results=False):
        """
        This method saves its results on the label name_inferred_label(label, node, radius, 'dlp')
        See documentation inference_models.py
        """
        # Avoid double counting
        node_set = self.to_set(node_set)
        radius_values = self.to_set(radius_values)

        # first element of list results counts prediction successes
        # second element of list results keeps sum of  distance from predictions to true value
        # third element of list results counts total nodes in which we have performed inference
        results = [0] * 3

        # perform inference
        for r in radius_values:
            if not isinstance(r, int) or r < 0:  # check r is a positive integer
                raise ValueError("Radius must be a positive integer")

            for v in node_set:
                if v not in self.graph.nodes:
                    raise ValueError("Node '" + v + "' does not exist in the graph")

                discrete_label_propagation(self, v, r, num_iterations, label)
                if count_results:
                    inferred_label = self.get_inferred_label(v, r, 'dlp', label)
                    true_label = self.get_true_label(v, r, label)
                    for node in inferred_label:
                        results[2] += 1
                        aux = abs(inferred_label[node] - true_label[node])
                        results[1] += aux
                        if aux == 0:
                            results[0] += 1

                self.clear_cache()
                if clear_results:
                    self.clear_inferred_opinions(v, r, label)

        if count_results:
            return results

    # ----------------------------------------------------
    # Metrics
    # ----------------------------------------------------

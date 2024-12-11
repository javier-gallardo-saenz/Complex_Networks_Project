import networkx as nx
import numpy as np
from tqdm import tqdm
import random


def assign_initial_opinions(G, states=[-1, 0, 1], probabilities=[0.4, 0.2, 0.4]):
    """
    Asigna opiniones iniciales a cada nodo del grafo basándose en probabilidades especificadas.

    Parámetros:
    - G (networkx.Graph): El grafo.
    - states (list): Estados de opinión posibles (por defecto: -1, 0, 1).
    - probabilities (list): Probabilidades correspondientes para cada estado.

    Retorna:
    - opinions (dict): Diccionario que mapea cada nodo a su opinión inicial.
    """
    opinions = {}
    for node in G.nodes():
        opinions[node] = np.random.choice(states, p=probabilities)
    return opinions


def basic_opinion_generator(G, states=[-1, 0, 1], probabilities=[0.4, 0.2, 0.4], num_steps=10000):
    """

    :param G: (Networkx graph) Graph
    :param states: (List) Possible opinions
    :param probabilities: (List) Probability of each opinion appearing in a node at time t=0
    :param num_steps: (Integer) Number of time-steps to run the generating model for
    :return: Opinion list for the graph
    """
    opinions = assign_initial_opinions(G, states, probabilities)

    for _ in tqdm(range(num_steps), desc="Generating initial opinion distribution"):
        node = random.choice(list(G.nodes()))
        neighbors = list(G.neighbors(node))
        if neighbors:
            neighbor = random.choice(neighbors)
            opinions[node] = opinions[neighbor]

    return opinions

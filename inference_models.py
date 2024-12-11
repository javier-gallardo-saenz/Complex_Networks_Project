def bayesian_inference(G, sampled_nodes, sampled_opinions, boundary, beta=1.0, max_iter=50, tol=1e-5):
    """
    Infers opinions using belief propagation (Bayesian inference).

    Parameters:
    - G (networkx.Graph): The graph.
    - sampled_nodes (set): Set of sampled node IDs.
    - sampled_opinions (dict): Dictionary of sampled nodes and their opinions.
    - boundary (set): Set of boundary node IDs.
    - beta (float): Interaction strength parameter.
    - max_iter (int): Maximum number of iterations.
    - tol (float): Convergence tolerance.

    Returns:
    - inferred_opinions (dict): Dictionary of inferred opinions for boundary nodes.
    """
    # Possible opinions
    possible_opinions = [-1, 0, 1]
    num_states = len(possible_opinions)

    # Mapping from opinion to index
    opinion_to_index = {opinion: idx for idx, opinion in enumerate(possible_opinions)}    # diccionario {-1:0, 0:1, 1:2}
    index_to_opinion = {idx: opinion for idx, opinion in enumerate(possible_opinions)}    # diccionario {0:-1, 1:0, 2:1}

    # Initialize messages: For each directed edge, store a message (array of size num_states)
    messages = {}
    for edge in G.edges():
        u, v = edge
        messages[(u, v)] = np.ones(num_states) / num_states    # asigna vector (1/3, 1/3, 1/3) a cada edge (porque hay 3 stages)
        messages[(v, u)] = np.ones(num_states) / num_states

    # For observed nodes, fix their messages
    observed_messages = {}
    for node in sampled_nodes:
        observed_opinion = sampled_opinions[node]
        fixed_message = np.zeros(num_states)
        fixed_message[opinion_to_index[observed_opinion]] = 1.0
        for neighbor in G.neighbors(node):
            messages[(node, neighbor)] = fixed_message.copy()
        observed_messages[node] = fixed_message.copy()

    # Iterative message passing
    for iteration in range(max_iter):
        delta = 0  # Change in messages for convergence check
        new_messages = {}
        for edge in G.edges():
            for direction in [(edge[0], edge[1]), (edge[1], edge[0])]:
                i, j = direction
                if i in sampled_nodes:
                    continue  # Messages from observed nodes are fixed
                # Compute the new message from i to j
                product = np.ones(num_states)
                for k in G.neighbors(i):
                    if k != j:
                        product *= messages[(k, i)]
                # Multiply by node potential (uniform for unobserved nodes)
                # Compute the message
                m_ij = np.zeros(num_states)
                for xi in possible_opinions:
                    idx_i = opinion_to_index[xi]
                    sum_over_xj = 0
                    for xj in possible_opinions:
                        idx_j = opinion_to_index[xj]
                        # Edge potential: favor same opinions
                        if xi == xj:
                            edge_potential = np.exp(beta)
                        else:
                            edge_potential = np.exp(-beta)
                        sum_over_xj += edge_potential * product[idx_i]
                    m_ij[idx_i] = sum_over_xj
                # Normalize message
                m_ij /= np.sum(m_ij)
                # Update delta
                delta += np.sum(np.abs(m_ij - messages[(i, j)]))
                new_messages[(i, j)] = m_ij
        # Update messages
        for key in new_messages:
            messages[key] = new_messages[key]
        # Check convergence
        if delta < tol:
            print(f"Belief propagation converged after {iteration + 1} iterations.")
            break
    else:
        print(f"Belief propagation did not converge after {max_iter} iterations.")

    # Compute marginals
    marginals = {}
    for node in G.nodes():
        if node in sampled_nodes:
            marginals[node] = observed_messages[node]
        else:
            # Compute the product of incoming messages
            incoming_messages = []
            for neighbor in G.neighbors(node):
                incoming_messages.append(messages[(neighbor, node)])
            marginal = np.ones(num_states)
            for msg in incoming_messages:
                marginal *= msg
            # Normalize
            marginal /= np.sum(marginal)
            marginals[node] = marginal

    # Infer opinions for boundary nodes
    inferred_opinions = {}
    for node in boundary:
        marginal = marginals[node]
        inferred_state_index = np.argmax(marginal)
        inferred_opinion = index_to_opinion[inferred_state_index]
        inferred_opinions[node] = inferred_opinion

    return inferred_opinions



def heuristic_based_inference(G, sampled_nodes, sampled_opinions, boundary):
    """
    Infers opinions using a heuristic-based method that considers node centrality.

    Parameters:
    - G (networkx.Graph): The graph.
    - sampled_nodes (set): Set of sampled node IDs.
    - sampled_opinions (dict): Dictionary of sampled nodes and their opinions.
    - boundary (set): Set of boundary node IDs.

    Returns:
    - inferred_opinions (dict): Dictionary of inferred opinions for boundary nodes.
    """
    inferred_opinions = {}

    # Compute PageRank as a centrality measure
    pagerank = nx.pagerank(G, alpha=0.85)

    for node in boundary:
        neighbors_in_S = set(G.neighbors(node)) & sampled_nodes
        if neighbors_in_S:
            # Aggregate opinions weighted by PageRank
            opinion_scores = {}
            for neighbor in neighbors_in_S:
                opinion = sampled_opinions[neighbor]
                weight = pagerank[neighbor]
                if opinion in opinion_scores:
                    opinion_scores[opinion] += weight
                else:
                    opinion_scores[opinion] = weight
            # Assign the opinion with the highest weighted score
            majority_opinion = max(opinion_scores.items(), key=lambda x: x[1])[0]
            inferred_opinions[node] = majority_opinion
        else:
            # Default to swing (0) if no sampled neighbors
            inferred_opinions[node] = 0
    return inferred_opinions


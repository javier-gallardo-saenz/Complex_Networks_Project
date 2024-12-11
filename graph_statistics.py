import matplotlib.pyplot as plt


# Obtain degree distribution of a graph and plot it
def degree_distribution(g):
    #g: graph
    degrees_g = [d for n, d in g.degree()]
    plt.figure(figsize=(8, 6))
    plt.hist(degrees_g, bins=range(min(degrees_g), max(degrees_g) + 1), edgecolor='black')
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()
    return degrees_g

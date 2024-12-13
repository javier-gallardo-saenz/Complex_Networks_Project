import matplotlib_inline as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter  # Para contar ocurrencias de opiniones

def evaluate_inference(inferred_opinions, true_opinions, boundary):
    """
    Evalúa el desempeño de la inferencia calculando el error total, error promedio y precisión.

    Parámetros:
    - inferred_opinions (dict): Opiniones inferidas para nodos en la frontera externa.
    - true_opinions (dict): Opiniones verdaderas de todos los nodos.
    - boundary (set): Conjunto de IDs de nodos en la frontera externa.

    Retorna:
    - total_error (int): Suma de las diferencias absolutas entre opiniones inferidas y verdaderas.
    - average_error (float): Error promedio por nodo en la frontera.
    - accuracy (float): Proporción de opiniones inferidas correctamente.
    """
    total_error = sum(abs(inferred_opinions[node] - true_opinions[node]) for node in boundary)
    accuracy = sum(1 for node in boundary if inferred_opinions[node] == true_opinions[node]) / len(boundary)
    average_error = total_error / len(boundary)
    return total_error, average_error, accuracy


def plot_confusion_matrix_custom(inferred_opinions, true_opinions, title):
    """
    Grafica la matriz de confusión para las opiniones inferidas.

    Parámetros:
    - inferred_opinions (dict): Opiniones inferidas para nodos en la frontera externa.
    - true_opinions (dict): Opiniones verdaderas de todos los nodos.
    - title (str): Título para el gráfico.
    """
    y_true = [true_opinions[node] for node in inferred_opinions.keys()]
    y_pred = [inferred_opinions[node] for node in inferred_opinions.keys()]
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Democrat (-1)', 'Swing (0)', 'Republican (1)'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

def plot_degree_distribution(G, title):
    """
    Grafica la distribución de grados de un grafo.

    Parámetros:
    - G (networkx.Graph): El grafo.
    - title (str): Título para el gráfico.
    """
    degrees = [d for n, d in G.degree()]
    plt.figure(figsize=(8, 6))
    sns.histplot(degrees, bins=50, kde=False, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Grado")
    plt.ylabel("Frecuencia")
    plt.show()

def plot_opinion_distribution(opinions, title):
    """
    Grafica la distribución de opiniones.

    Parámetros:
    - opinions (dict): Diccionario de opiniones de nodos.
    - title (str): Título para el gráfico.
    """
    opinion_counts = Counter(opinions.values())
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=['Democrat (-1)', 'Swing (0)', 'Republican (1)'],
        y=[opinion_counts.get(-1, 0), opinion_counts.get(0, 0), opinion_counts.get(1, 0)],
        palette=['blue', 'grey', 'red']
    )
    plt.title(title)
    plt.xlabel("Opinión")
    plt.ylabel("Cantidad")
    plt.show()

import numpy as np

"""
K-Means-Algorithmus, der die Cluster-Zentren und Labels zurückgibt.

X : numpy.ndarray
    Die Datenpunkte als 2D-Array (n_samples x n_features).
k : int
    Die Anzahl der Cluster.
max_iters : int, optional (default=100)
    Die maximale Anzahl der Iterationen.
tol : float, optional (default=1e-4)
    Toleranz, um die Konvergenz zu überprüfen (d.h. wann sich die Cluster-Zentren nicht mehr ändern).

Returns
centroids : numpy.ndarray
    Die finalen Cluster-Zentren.
labels : numpy.ndarray
    Die Labels für jeden Datenpunkt, das Cluster, dem sie zugewiesen wurden.
"""

def k_means(X, k, max_iters=100, tol=1e-4):
    n_samples = X.shape[0]
    # 1. Initialisiere zufällige Cluster-Zentren
    random_indices = np.random.choice(n_samples, size=k, replace=False)
    centroids = X[random_indices]
    for _ in range(max_iters):
        # 2. Berechne die Distanzen zu den Zentren und weise Cluster zu
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # Abstand jedes Punktes zu jedem Zentrum
        labels = np.argmin(distances, axis=1)  # Weist jedem Punkt das nächstgelegene Zentrum zu
        # 3. Berechne die neuen Cluster-Zentren als Mittelwert der zugewiesenen Punkte
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        # 4. Überprüfe, ob sich die Zentren geändert haben (Konvergenz)
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    return centroids, labels


if __name__ == '__main__':
    # Beispiel mit zufälligen 2D-Daten
    np.random.seed(42)
    X = np.random.rand(100, 2)  # 100 zufällige 2D-Punkte
    k = 3  # Anzahl der Cluster

    centroids, labels = k_means(X, k)

    # Ausgabe der finalen Cluster-Zentren und Labels
    print("Finale Cluster-Zentren:\n", centroids)
    print("Labels der Punkte:\n", labels)



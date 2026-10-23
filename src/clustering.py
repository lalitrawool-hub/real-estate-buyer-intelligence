"""Buyer clustering experiments and internal cluster quality metrics."""
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def cluster_buyers(frame: pd.DataFrame, preprocessor, method="kmeans", n_clusters=4, random_state=42):
    if len(frame) < 2:
        raise ValueError("At least two buyers are needed to fit a clustering model.")
    if not 2 <= int(n_clusters) < len(frame):
        raise ValueError("Choose a cluster count from 2 up to one less than the buyer count.")
    matrix = preprocessor.fit_transform(frame)
    if matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError("Preprocessing did not produce a finite feature matrix.")
    model_name = method.lower()
    if model_name == "kmeans":
        model = KMeans(n_clusters=int(n_clusters), random_state=random_state, n_init=10)
        labels = model.fit_predict(matrix)
    elif model_name in ("hierarchical", "agglomerative"):
        model = AgglomerativeClustering(n_clusters=int(n_clusters), linkage="ward")
        labels = model.fit_predict(matrix)
    else:
        raise ValueError("method must be 'kmeans' or 'hierarchical'.")
    return labels, model, matrix


def evaluate_clusters(matrix, labels):
    """Return silhouette (higher is better) and Davies–Bouldin (lower is better)."""
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return {"silhouette": np.nan, "davies_bouldin": np.nan}
    return {"silhouette": float(silhouette_score(matrix, labels)),
            "davies_bouldin": float(davies_bouldin_score(matrix, labels))}


def compare_methods(frame, preprocessor_factory, cluster_counts=(2, 3, 4, 5), random_state=42):
    """Run K-Means and hierarchical experiments across feasible cluster counts."""
    results = []
    for count in cluster_counts:
        if count >= len(frame):
            continue
        for method in ("kmeans", "hierarchical"):
            try:
                labels, _, matrix = cluster_buyers(frame, preprocessor_factory(), method, count, random_state)
                results.append({"method": method, "n_clusters": count, **evaluate_clusters(matrix, labels)})
            except (ValueError, TypeError):
                continue
    return pd.DataFrame(results)

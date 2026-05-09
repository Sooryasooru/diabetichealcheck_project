"""Clustering module."""

from sklearn.cluster import KMeans


def train_kmeans(x_scaled):
    """Train KMeans clustering."""

    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(x_scaled)

    return model, labels

"""Evaluation module."""

from sklearn.metrics import silhouette_score


def evaluate_model(x_scaled, labels):
    """Evaluate clustering performance."""

    score = silhouette_score(
        x_scaled,
        labels,
    )

    return {"silhouette_score": round(score, 3)}

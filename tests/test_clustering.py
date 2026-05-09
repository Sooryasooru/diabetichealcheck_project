import numpy as np

from src.clustering import train_kmeans


def test_train_kmeans():
    data = np.random.rand(10, 4)

    model, labels = train_kmeans(data)

    assert len(labels) == 10

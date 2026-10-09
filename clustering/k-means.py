import numpy as np

from .lib import (
    TInputFeatures,
    TTrainFeatures,
    datasets,
    input_features,
    train_features,
)

TCentroids = TTrainFeatures


def label(M: TCentroids, X: TTrainFeatures):
    return ((X[:, None, :] - M[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)


def train(X: TTrainFeatures, M: TCentroids) -> TCentroids:
    k = len(M)

    while True:
        Y = label(M, X)

        # update
        sums = np.zeros((k, X.shape[1]))
        counts = np.zeros(k)
        np.add.at(sums, Y, X)  # sums[Y[i]] += X[i]
        np.add.at(counts, Y, 1)  # counts[Y[i]] += 1

        keep_clusters = counts > 0
        if keep_clusters.all():
            next_M = sums / counts[:, None]
            # whether M converges
            if np.allclose(M, next_M):
                break
            M = next_M
        else:
            M = sums[keep_clusters] / counts[keep_clusters, None]
            k = len(M)

    return M


def initial_centroids(
    k: int, X: TTrainFeatures, rng=np.random.default_rng(42)
) -> TCentroids:
    """choose initial centroids using K-means++"""
    N = X.shape[0]

    M = np.empty((k, X.shape[1]))
    M[0] = X[rng.integers(N)]

    distances = ((X - M[0]) ** 2).sum(axis=1)
    for j in range(1, k):
        m = X[rng.choice(N, p=distances / distances.sum())]
        M[j] = m
        np.minimum(distances, ((X - m) ** 2).sum(axis=1), out=distances)
    return M


def predict(M: TCentroids, x: TInputFeatures):
    return ((M - x) ** 2).sum(axis=1).argmin()


if __name__ == "__main__":
    X = train_features(datasets)

    M = train(X, initial_centroids(3, X))
    print("centroids:", M)
    print("clusters:", len(M))

    x = input_features(input("x y: "))
    print("input features:", x)

    print("predicted cluster:", predict(M, x))

import numpy as np
from numpy import float64

from .lib import Matrix, input_features, train_expectations, train_features
from .lib import datasets_with_noise as datasets


def train(X: Matrix, Y: Matrix, regularization: float64) -> Matrix:
    I = np.eye(X.shape[1], dtype=float64)
    I[0, 0] = 0  # ignore intercept

    return np.linalg.solve(
        X.T @ X + regularization * I,
        X.T @ Y,
    )


def error(w: Matrix, X: Matrix, Y: Matrix) -> float64:
    return np.mean((Y - X @ w) ** 2)


def predict(w: Matrix, x: Matrix):
    return x @ w


def k_fold(
    k: int,
    X: Matrix,
    Y: Matrix,
    regularization: float64,
    seed: int = 42,
) -> float64:
    indices = np.arange(len(X))
    np.random.default_rng(seed).shuffle(indices)
    folds = np.array_split(indices, k)

    errors = float64(0)
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        errors += error(
            train(X[train_idx], Y[train_idx], regularization), X[test_idx], Y[test_idx]
        )

    return errors / k


def select_regularization(
    k: int, X: Matrix, Y: Matrix, regularizations: Matrix
) -> float64:
    return regularizations[
        np.argmin(
            [k_fold(k, X, Y, regularization) for regularization in regularizations]
        )
    ]


if __name__ == "__main__":
    X = train_features(datasets)
    # X -> log(Y)
    Y = np.log(train_expectations(datasets))

    w = train(
        X,
        Y,
        select_regularization(
            5, X, Y, np.array([0.01, 0.1, 1, 2, 5, 10, 100, 1000], dtype=float64)
        ),
    )
    print("weights:", w)
    print("error:", np.exp(error(w, X, Y)))

    x = input_features(input("area (m^2): "), input("distance from city center (km): "))
    print("input features:", x)
    print("predicted price:", np.exp(predict(w, x)))

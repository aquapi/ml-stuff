import numpy as np
from numpy import float64

from .lib import (
    TInputFeatures,
    TTrainExpectation,
    TTrainFeatures,
    input_features,
    train_expectations,
    train_features,
)
from .lib import datasets_with_noise as datasets

TWeights = np.ndarray[tuple[int], np.dtype[np.float64]]


def train(X: TTrainFeatures, y: TTrainExpectation, regularization: float) -> TWeights:
    I = np.eye(X.shape[1], dtype=float64)
    I[0, 0] = 0  # ignore intercept

    return np.linalg.solve(
        X.T @ X + regularization * I,
        X.T @ y,
    )


def error(w: TWeights, X: TTrainFeatures, y: TTrainExpectation) -> float64:
    return np.mean((y - X @ w) ** 2)


def predict(w: TWeights, x: TInputFeatures):
    return x @ w


def k_fold(
    k: int,
    X: TTrainFeatures,
    y: TTrainExpectation,
    regularization: float,
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
            train(X[train_idx], y[train_idx], regularization), X[test_idx], y[test_idx]
        )

    return errors / k


def select_regularization(
    k: int, X: TTrainFeatures, y: TTrainExpectation, regularizations: list[float]
) -> float:
    return regularizations[
        np.argmin(
            [k_fold(k, X, y, regularization) for regularization in regularizations]
        )
    ]


if __name__ == "__main__":
    X = train_features(datasets)
    # X -> log(y)
    y = np.log(train_expectations(datasets))

    w = train(
        X,
        y,
        select_regularization(5, X, y, [0.01, 0.1, 1, 2, 5, 10, 100, 1000]),
    )
    print("weights:", w)
    print("error:", np.exp(error(w, X, y)))

    x = input_features(input("area (m^2): "), input("distance from city center (km): "))
    print("input features:", x)
    print("predicted price:", np.exp(predict(w, x)))

import numpy as np
from numpy import float64

from .lib import (
    TInputFeatures,
    TTrainExpectation,
    TTrainFeatures,
    datasets,
    input_features,
    train_expectations,
    train_features,
)

TWeights = np.ndarray[tuple[int], np.dtype[np.float64]]


def train(X: TTrainFeatures, y: TTrainExpectation) -> TWeights:
    return np.linalg.lstsq(X, y)[0]


def error(w: TWeights, X: TTrainFeatures, y: TTrainExpectation) -> float64:
    return ((y - X @ w) ** 2).mean()


def predict(w: TWeights, x: TInputFeatures):
    return x @ w


if __name__ == "__main__":
    X = train_features(datasets)
    # X -> log(y)
    y = np.log(train_expectations(datasets))

    w = train(X, y)
    print("weights:", w)
    print("error:", np.exp(error(w, X, y)))

    x = input_features(input("area (m^2): "), input("distance from city center (km): "))
    print("input features:", x)
    print("predicted price:", np.exp(predict(w, x)))

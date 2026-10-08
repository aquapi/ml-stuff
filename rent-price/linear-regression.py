import numpy as np
from numpy import float64

from .lib import Matrix, datasets, input_features, train_expectations, train_features


def train(X: Matrix, Y: Matrix) -> Matrix:
    return np.linalg.lstsq(X, Y)[0]


def error(w: Matrix, X: Matrix, Y: Matrix) -> float64:
    return np.mean((Y - X @ w) ** 2)


def predict(w: Matrix, x: Matrix):
    return x @ w


if __name__ == "__main__":
    X = train_features(datasets)
    # X -> log(Y)
    Y = np.log(train_expectations(datasets))

    w = train(X, Y)
    print("weights:", w)
    print("error:", np.exp(error(w, X, Y)))

    print(
        "predicted price:",
        np.exp(
            predict(
                w,
                input_features(
                    input("area (m^2): "), input("distance from city center (km): ")
                ),
            )
        ),
    )

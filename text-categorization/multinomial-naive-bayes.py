import numpy as np
from numpy.typing import NDArray

from .lib import bow_input_features, bow_train_features
from .lib import simple_datasets as datasets


def train(X: NDArray[np.intp], lsp: float = 1):
    """
    # Parameters
    X[c] -> [n(c), ...n(x|c)]
    lsp: laplace smoothing parameter
    """

    datasets_len = X[:, 0].sum()
    bow_size = X.shape[1] - 1

    W = np.zeros(X.shape, dtype=np.float64)

    for i, x in enumerate(X):
        W[i] = (x + lsp) / (np.sum(x[1:]) + bow_size * lsp)
        W[i, 0] = x[0] / datasets_len

    return np.log(W)


def predict(W: NDArray[np.float64], x: NDArray[np.intp]):
    return np.argmax(W @ x)


if __name__ == "__main__":
    (bow, X) = bow_train_features(datasets)
    print("bag of words:", bow)

    W = train(X)
    print("weights:", W)

    x = bow_input_features(bow, input("text: "))
    print("input features:", x)
    print("label:", predict(W, x))

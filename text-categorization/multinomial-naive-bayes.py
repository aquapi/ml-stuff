import numpy as np
from numpy.typing import NDArray

from .lib import bow_input_features, bow_train_features
from .lib import simple_datasets as datasets


def train(X: NDArray[np.intp], datasets_len: int, lsc: float = 1):
    """
    # Parameters
    X[c] -> [n(c), ...n(x|c)]
    datasets_len: dataset count
    lsc: laplace smoothing parameter
    """
    w = np.zeros(X.shape, dtype=np.float64)
    bow_size = X.shape[1] - 1

    for i, x in enumerate(X):
        nc = x[0]

        w[i] = (x + lsc) / (np.sum(x[1:]) + bow_size * lsc)  # laplace smoothing
        w[i, 0] = nc / datasets_len

    return np.log(w)


def predict(w: NDArray[np.float64], x: NDArray[np.intp]):
    return np.argmax(w @ x)


if __name__ == "__main__":
    (bow, X) = bow_train_features(datasets)
    print("bag of words:", bow)

    w = train(X, len(datasets))
    print("weights:", w)

    x = bow_input_features(bow, input("text: "))
    print("label:", predict(w, x))

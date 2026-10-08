import numpy as np

from .lib import Matrix, bow_input_features, bow_train_features
from .lib import simple_datasets as datasets


def train(X: Matrix, datasets_len: int, lsc: float = 1):
    bow_size = X.shape[1] - 1

    for i in range(len(X)):
        x = X[i]

        nc = x[0]  # datasets count with label c
        x = (x + lsc) / (np.sum(x) - nc + bow_size * lsc)  # laplace smoothing
        x[0] = nc / datasets_len

        X[i] = x

    return np.log(X)


def predict(w: Matrix, x: Matrix):
    return np.argmax(w @ x)


if __name__ == "__main__":
    (bow, X) = bow_train_features(datasets)
    w = train(X, len(datasets))

    print("weights:", w)

    x = bow_input_features(bow, input("text: "))
    print(predict(w, x))

import numpy as np

from .lib import TInputFeatures, TTrainFeatures, bow_input_features, bow_train_features
from .lib import simple_datasets as datasets


def train(W: TTrainFeatures, lsp: float = 1):
    """
    # Parameters
    X[c] -> [n(c), ...n(x|c)]
    lsp: laplace smoothing parameter
    """

    datasets_len = W[:, 0].sum()
    bow_size = W.shape[1] - 1

    for i, x in enumerate(W):
        pc = x[0] / datasets_len
        W[i] = (x + lsp) / (x[1:].sum() + bow_size * lsp)
        W[i, 0] = pc

    np.log(W, out=W)


def predict(W: TTrainFeatures, x: TInputFeatures):
    return (W @ x).argmax()


if __name__ == "__main__":
    bow, W = bow_train_features(datasets)
    print("bag of words:", bow)

    train(W)
    print("weights:", W)

    x = bow_input_features(bow, input("text: "))
    print("input features:", x)
    print("label:", predict(W, x))

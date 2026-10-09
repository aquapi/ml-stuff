import numpy as np

TDatasets = list[tuple[str, int]]
TBoW = dict[str, int]
TTrainFeatures = np.ndarray[tuple[int, int], np.dtype[np.float64]]
TInputFeatures = np.ndarray[tuple[int], np.dtype[np.float64]]

simple_datasets: TDatasets = [
    ("hanoi pho chaolong hanoi", 0),
    ("hanoi buncha pho omai", 0),
    ("pho banhgio omai", 0),
    ("saigon hutiu banhbo pho", 1),
]


def bow_train_features(datasets: TDatasets) -> tuple[TBoW, TTrainFeatures]:
    """
    # Returns
    bow[word] -> index in feature vector

    X[c] -> [n(c), ...n(x|c)]
    """
    bow: TBoW = {}
    bow_size = 0
    max_label = 0

    # setup vocabulary
    for dataset in datasets:
        for word in dataset[0].split():
            if word not in bow:
                bow_size += 1
                bow[word] = bow_size

        max_label = max(max_label, dataset[1])

    # feature vectors: c -> [n(c), ...n(x[i]|c)]
    X = np.zeros((max_label + 1, bow_size + 1), dtype=np.float64)

    # count words of each label
    for dataset in datasets:
        x = X[dataset[1]]
        x[0] += 1
        for word in dataset[0].split():
            x[bow[word]] += 1

    return bow, X


def bow_input_features(bow: TBoW, s: str) -> TInputFeatures:
    """
    # Returns
    [1, ...BoW word counts]
    """
    x = np.zeros(len(bow) + 1, dtype=np.float64)
    x[0] = 1

    for word in s.split():
        if word in bow:
            x[bow[word]] += 1

    return x

import numpy as np

simple_datasets = [
    ("hanoi pho chaolong hanoi", 0),
    ("hanoi buncha pho omai", 0),
    ("pho banhgio omai", 0),
    ("saigon hutiu banhbo pho", 1),
]


def bow_train_features(datasets: list[tuple[str, int]]):
    """
    # Returns
    bow[word] -> index in feature vector

    X[c] -> [n(c), ...n(x|c)]
    """
    bow: dict[str, int] = {}
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
    X = np.zeros((max_label + 1, bow_size + 1), dtype=np.intp)

    # count words of each label
    for dataset in datasets:
        x = X[dataset[1]]
        x[0] += 1
        for word in dataset[0].split():
            x[bow[word]] += 1

    return (bow, X)


def bow_input_features(bow: dict[str, int], s: str):
    """
    # Returns
    [1, ...BoW word counts]
    """
    x = np.zeros(len(bow) + 1, dtype=np.intp)
    x[0] = 1

    for word in s.split():
        if word in bow:
            x[bow[word]] += 1

    return x

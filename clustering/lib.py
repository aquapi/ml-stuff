import numpy as np

TDatasets = list[tuple[float, float]]
TTrainFeatures = np.ndarray[tuple[int, int], np.dtype[np.float64]]
TInputFeatures = np.ndarray[tuple[int], np.dtype[np.float64]]

datasets: TDatasets = [
    (1.0, 1.0),
    (1.2, 1.1),
    (0.8, 1.3),
    (1.1, 0.8),
    (0.9, 0.9),
    (1.3, 1.4),
    (0.7, 0.7),
    (1.4, 1.0),
    (1.0, 1.5),
    (0.6, 1.2),
    (1.5, 1.2),
    (1.1, 1.6),
    (0.8, 0.6),
    (1.2, 0.7),
    (0.5, 0.9),
    (1.6, 0.9),
    (1.0, 0.5),
    (5.0, 5.0),
    (5.2, 5.1),
    (4.8, 5.3),
    (5.1, 4.8),
    (4.9, 4.9),
    (5.3, 5.4),
    (4.7, 4.7),
    (5.4, 5.0),
    (5.0, 5.5),
    (4.6, 5.2),
    (5.5, 5.2),
    (5.1, 5.6),
    (4.8, 4.6),
    (5.2, 4.7),
    (4.5, 4.9),
    (5.6, 4.9),
    (9.0, 1.0),
    (9.2, 1.1),
    (8.8, 1.3),
    (9.1, 0.8),
    (8.9, 0.9),
    (9.3, 1.4),
    (8.7, 0.7),
    (9.4, 1.0),
    (9.0, 1.5),
    (8.6, 1.2),
    (9.5, 1.2),
    (9.1, 1.6),
    (8.8, 0.6),
    (9.2, 0.7),
    (8.5, 0.9),
    (9.6, 0.9),
    (9.0, 0.5),
]

def train_features(datasets: TDatasets) -> TTrainFeatures:
  return np.array(datasets, dtype=np.float64)


def input_features(pair: str) -> TInputFeatures:
  return np.array(pair.split(), dtype=np.float64)

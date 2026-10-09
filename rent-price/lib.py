from typing import Any

import numpy as np

TDatasets = list[tuple[tuple[int, int], int]]
TTrainFeatures = np.ndarray[tuple[int, int], np.dtype[np.float64]]
TTrainExpectation = np.ndarray[tuple[int], np.dtype[np.float64]]
TInputFeatures = np.ndarray[tuple[int], np.dtype[np.float64]]

datasets: TDatasets = [
    ((20, 5), 6),
    ((30, 4), 10),
    ((17, 6), 3),
    ((25, 4), 8),
    ((27, 6), 7),
    ((23, 8), 4),
    ((19, 6), 5),
    ((35, 3), 13),
    ((42, 2), 17),
    ((28, 7), 6),
    ((50, 5), 14),
    ((32, 9), 7),
    ((45, 4), 15),
    ((22, 10), 4),
    ((38, 6), 11),
    ((55, 3), 19),
    ((26, 5), 7),
    ((60, 8), 14),
    ((33, 2), 12),
    ((48, 7), 12),
]

datasets_with_noise = datasets.copy()
datasets_with_noise.extend(
    [
        # unusual data
        ((65, 3), 8),
        ((12, 2), 15),
        ((25, 12), 15),
        ((70, 1), 5),
    ]
)


def train_features(datasets: TDatasets) -> TTrainFeatures:
    X = np.array([d[0] for d in datasets], dtype=np.float64)

    areas = X[:, 0]
    distances = X[:, 1]

    return np.column_stack(
        [
            np.ones(len(X)),  # intercept
            areas,
            distances**2,
        ]
    )


def train_expectations(datasets: TDatasets) -> TTrainExpectation:
    return np.array([d[1] for d in datasets], dtype=np.float64)


def input_features(area: Any, distance: Any) -> TInputFeatures:
    return np.array(
        [
            np.float64(1),  # intercept
            np.float64(area),
            np.float64(distance) ** 2,
        ]
    )

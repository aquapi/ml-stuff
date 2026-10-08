from typing import Any

import numpy as np
from numpy import float64
from numpy.typing import NDArray

Matrix = NDArray[float64]

datasets = [
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


def train_features(datasets: list[Any]) -> Matrix:
    X = np.array([float64(d[0]) for d in datasets])

    areas = X[:, 0]
    distances = X[:, 1]

    return np.column_stack(
        [
            np.ones(len(X)),  # intercept
            areas,
            distances**2,
        ]
    )


def train_expectations(datasets: list[Any]) -> Matrix:
    return np.array([float64(d[1]) for d in datasets])


def input_features(area: Any, distance: Any) -> Matrix:
    return np.array(
        [
            float64(1),  # intercept
            float64(area),
            float64(distance) ** 2,
        ]
    )

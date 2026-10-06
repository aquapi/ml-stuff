import numpy as np
from numpy.typing import NDArray


def cross_validation_train(
    test_datasets: int,
    X: NDArray,
    Y: NDArray,
    X_train: list = [],  # noqa: B006
    Y_train: list = [],  # noqa: B006
    X_test: list = [],  # noqa: B006
    Y_test: list = [],  # noqa: B006
    start_idx: int = 0,
) -> tuple[NDArray, float, float]:
    """Pick a specified number of datasets for testing, use others for training, choose the best weights based on their training errors and test errors"""

    if test_datasets == 0:
        pop_cnt = len(X) - start_idx
        while start_idx < len(X):
            # Add remaining datasets to training sets
            X_train.append(X[start_idx])
            Y_train.append(Y[start_idx])

            start_idx += 1

        X_train_array = np.array(X_train)
        Y_train_array = np.array(Y_train)

        w = train(X_train_array, Y_train_array)

        # Reset training set
        while pop_cnt > 0:
            X_train.pop()
            Y_train.pop()

            pop_cnt -= 1

        return (
            w,
            # Training cost
            cost(w, X_train_array, Y_train_array),
            # Test cost
            cost(w, np.array(X_test), np.array(Y_test)),
        )

    remaining_tests = test_datasets - 1

    # First case
    X_test.append(X[start_idx])
    Y_test.append(Y[start_idx])
    optimal_pair = cross_validation_train(
        remaining_tests, X, Y, X_train, Y_train, X_test, Y_test, start_idx + 1
    )

    cur_idx = start_idx + 1
    pop_cnt = len(X) - test_datasets - cur_idx
    while cur_idx < len(X) - test_datasets:
        # Swap the current set as test set
        X_test[-1] = X[cur_idx]
        Y_test[-1] = Y[cur_idx]

        # Add previous case as a training set
        X_train.append(X[cur_idx - 1])
        Y_train.append(Y[cur_idx - 1])

        # Pick weights
        result_pair = cross_validation_train(
            remaining_tests,
            X,
            Y,
            X_train,
            Y_train,
            X_test,
            Y_test,
            cur_idx + 1,
        )

        # TODO: better selection maybe?
        if result_pair[1] < optimal_pair[1] and result_pair[2] <= optimal_pair[2]:
            optimal_pair = result_pair

        cur_idx += 1

    # Reset testing set
    X_test.pop()
    Y_test.pop()

    # Reset training set
    while pop_cnt > 0:
        X_train.pop()
        Y_train.pop()

        pop_cnt -= 1

    return optimal_pair


def train(X: NDArray, Y: NDArray):
    return np.linalg.multi_dot([np.linalg.pinv(np.dot(X.T, X)), X.T, Y])


def evaluate(x: NDArray, w: NDArray):
    return np.inner(x, w)


def cost(w: NDArray, X: NDArray, Y: NDArray) -> float:
    return np.linalg.norm(Y - X.dot(w), 2) / len(w)

# Predict rent price of a house based on area and distance from city center
datasets = [
    [[20, 5], 6],
    [[30, 4], 10],
    [[17, 6], 3],
    [[25, 4], 8],
    [[27, 6], 7],
    [[23, 8], 4],
    [[19, 6], 5],
    [[35, 3], 13],
    [[42, 2], 17],
    [[28, 7], 6],
    [[50, 5], 14],
    [[32, 9], 7],
    [[45, 4], 15],
    [[22, 10], 4],
    [[38, 6], 11],
    [[55, 3], 19],
    [[26, 5], 7],
    [[60, 8], 14],
    [[33, 2], 12],
    [[48, 7], 12],
]
X = np.array([d[0] for d in datasets])
Y = np.array([d[1] for d in datasets])

(w, training_cost, test_cost) = cross_validation_train(4, X, Y)
print("training cost:", training_cost)
print("test cost:", test_cost)

x = np.array(
    [float(input("area (m^2): ")), float(input("distance from city center (km): "))]
)
print("predicted price:", evaluate(x, w))

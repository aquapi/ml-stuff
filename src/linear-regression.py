import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float64]

def train(X: Matrix, Y: Matrix):
    """X @ w ~ Y"""
    return np.linalg.lstsq(X, Y)[0]

def mse(w: Matrix, X: Matrix, Y: Matrix):
    return np.mean((Y - X @ w) ** 2)

def evaluate(x: Matrix, w: Matrix):
    return x @ w

def k_fold(k: int, X: Matrix, Y: Matrix, seed: int | None = None):
  indices = np.arange(len(Y))
  np.random.default_rng(seed).shuffle(indices)

  folds = np.array_split(indices, k)

  total_error = np.float64(0)

  for i in range(k):
    train_indices = np.concat([folds[j] for j in range(k) if j != i])
    w = train(X[train_indices], Y[train_indices])

    test_indices = folds[i];
    total_error += mse(w, X[test_indices], Y[test_indices])

  return total_error / k

if __name__ == '__main__':
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
    X = np.array([d[0] for d in datasets], dtype=np.float64)
    Y = np.array([d[1] for d in datasets], dtype=np.float64)

    w = train(X, Y)
    print('mse:', mse(w, X, Y))
    print('k-fold mse:', k_fold(5, X, Y))

    x = np.array(
        [float(input("area (m^2): ")), float(input("distance from city center (km): "))],
        dtype=np.float64
    )
    print("predicted price:", evaluate(x, w))

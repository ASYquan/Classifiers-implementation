import numpy as np
from typing import Iterable, Tuple
from utils import add_bias, accuracy

class NumpyClassifier:
    """Base class placeholder (for parity with original structure)."""

class NumpyLinRegClass(NumpyClassifier):
    """
    Linear regression used as a classifier with MSE loss and SGD over epochs.
    """
    def __init__(self, bias: float = -1.0):
        self.bias = bias
        self.weights = None

    def fit(self, X_train: np.ndarray, t_train: np.ndarray,
            lr: float = 0.1, epochs: int = 10):
        if self.bias is not None:
            Xb = add_bias(X_train, self.bias)
        else:
            Xb = X_train
        N, M = Xb.shape
        self.weights = np.zeros(M)
        for _ in range(epochs):
            grad = (Xb.T @ (Xb @ self.weights - t_train)) / N
            self.weights -= lr * grad

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        Xb = add_bias(X, self.bias) if self.bias is not None else X
        ys = Xb @ self.weights
        return (ys > threshold).astype(int)

def grid_search_linreg(
    X_train: np.ndarray, t_train: np.ndarray,
    X_val: np.ndarray, t_val: np.ndarray,
    lrs: Iterable[float], epochs_list: Iterable[int]
) -> Tuple[Tuple[float, int, float], list]:
    """
    Returns (best_lr, best_epochs, best_acc), and a list of (lr, epochs, acc)
    """
    results = []
    best = (None, None, -np.inf)
    for lr in lrs:
        for epochs in epochs_list:
            cl = NumpyLinRegClass()
            cl.fit(X_train, t_train, lr=lr, epochs=epochs)
            acc = accuracy(cl.predict(X_val), t_val)
            results.append((lr, epochs, acc))
            if acc > best[2]:
                best = (lr, epochs, acc)
    return best, results

import numpy as np
import matplotlib.pyplot as plt

def add_bias(X: np.ndarray, bias: float):
    """
    X: (N, M)
    bias: scalar (e.g., -1 or 1). Use 0 for 'no-bias' behavior (still adds a column).
    Returns: (N, M+1) with bias column in position 0.
    """
    N = X.shape[0]
    biases = np.ones((N, 1)) * bias
    return np.concatenate((biases, X), axis=1)

def accuracy(predicted: np.ndarray, gold: np.ndarray) -> float:
    return float(np.mean(predicted == gold))

def standard_scaler(X_train: np.ndarray, X_other: np.ndarray):
    """
    Fit scaler on X_train and transform both X_train and X_other.
    Returns: X_train_scaled, X_other_scaled
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std = np.where(std == 0, 1.0, std)  # guard against zero std
    return (X_train - mean) / std, (X_other - mean) / std

def plot_decision_regions(X: np.ndarray, t: np.ndarray, clf, size=(8, 6), cmap="tab10"):
    """
    Plot the data (X, t) with decision regions for classifier clf.
    clf must implement predict(X) -> labels.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid).reshape(xx.shape)

    plt.figure(figsize=size)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], c=t, s=10.0, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision regions")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.show()

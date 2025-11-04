import numpy as np
from typing import Optional, Dict, List
from utils import add_bias, accuracy

class NumpyClassifier:
    pass

class NumpyLogReg(NumpyClassifier):
    """
    Binary logistic regression with early stopping, tracking loss/accuracy and optional val.
    """
    def __init__(self, bias: float = -1.0):
        self.bias = bias
        self.weights = None
        self.n_epochs = 0
        self.losses: Dict[str, List[float]] = {}
        self.accs: Dict[str, List[float]] = {}

    @staticmethod
    def _sigmoid(x):
        # Numerically stable sigmoid
        z = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _forward_logits(self, Xb):
        return Xb @ self.weights

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Returns p(y=1|x) shape (N,), adding bias if needed.
        """
        if X.shape[1] == (self.weights.shape[0] - 1):
            Xb = add_bias(X, self.bias)
        elif X.shape[1] == self.weights.shape[0]:
            Xb = X
        else:
            raise ValueError("Input has incorrect number of columns for current weights.")
        return self._sigmoid(self._forward_logits(Xb))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        p = self.predict_probabilities(X)
        return (p > threshold).astype(int)

    def binary_cross_entropy(self, X: np.ndarray, t: np.ndarray) -> float:
        p = self.predict_probabilities(X)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))

    def fit(
        self,
        X_train: np.ndarray,
        t_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        t_val: Optional[np.ndarray] = None,
        eta: float = 0.1,
        epochs: int = 100,
        tol: float = 0.0,
        n_epochs_no_update: int = 2
    ):
        self.losses = {"train": [], "val": []}
        self.accs = {"train": [], "val": []}
        self.n_epochs = 0

        Xb_train = add_bias(X_train, self.bias)
        N, M = Xb_train.shape
        self.weights = np.zeros(M)

        best_loss = np.inf
        epochs_no_improve = 0

        for e in range(epochs):
            self.n_epochs += 1

            logits = Xb_train @ self.weights
            p = self._sigmoid(logits)
            # Gradient of BCE wrt weights
            grad = (Xb_train.T @ (p - t_train)) / N
            self.weights -= eta * grad

            # Track train loss/acc
            train_loss = self.binary_cross_entropy(X_train, t_train)
            self.losses["train"].append(train_loss)
            self.accs["train"].append(accuracy(self.predict(X_train), t_train))

            # Track val if provided
            if X_val is not None and t_val is not None:
                val_loss = self.binary_cross_entropy(X_val, t_val)
                self.losses["val"].append(val_loss)
                self.accs["val"].append(accuracy(self.predict(X_val), t_val))

            # Early stopping on training loss
            if train_loss < best_loss - tol:
                best_loss = train_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= n_epochs_no_update:
                    break

def grid_search_logreg(
    X_train, t_train, X_val, t_val,
    eta_list=(1, 0.1, 0.01, 0.001, 0.0001, 0.00001),
    epochs_list=(1, 2, 5, 10, 50, 100, 300, 1000),
    tol_list=(1e-3, 1e-4, 1e-5)
):
    results = []
    best = None
    best_acc = -np.inf
    for epochs in epochs_list:
        for eta in eta_list:
            for tol in tol_list:
                model = NumpyLogReg()
                model.fit(X_train, t_train, X_val=X_val, t_val=t_val,
                          eta=eta, epochs=epochs, tol=tol)
                acc = accuracy(model.predict(X_val), t_val)
                results.append((epochs, eta, tol, acc))
                if acc > best_acc:
                    best_acc = acc
                    best = (epochs, eta, tol, acc, model)
    return best, results

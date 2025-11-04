import numpy as np
from typing import Optional, Dict, List
from utils import add_bias, accuracy

def logistic(x):
    z = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def logistic_diff(y):
    # y is already sigmoid(x)
    return y * (1.0 - y)

class MLPBinaryLinRegClass:
    """
    MLP with one hidden layer (logistic activation), linear output.
    Trained with MSE on the linear output (as per original assignment baseline).
    Tracks loss/accuracy and supports early stopping.
    """
    def __init__(self, bias: float = -1.0, dim_hidden: int = 6):
        self.bias = bias
        self.dim_hidden = dim_hidden
        self.activ = logistic
        self.activ_diff = logistic_diff
        self.weights1 = None
        self.weights2 = None
        self.n_epochs = 0
        self.losses: Dict[str, List[float]] = {}
        self.accs: Dict[str, List[float]] = {}

    def forward(self, Xb: np.ndarray):
        """
        Forward pass.
        Xb: (N, dim_in+1) with bias already included
        Returns:
          hidden_outs: (N, dim_hidden+1) with bias included
          outputs: (N, 1) linear outputs
        """
        hidden_acts = self.activ(Xb @ self.weights1)               # (N, H)
        hidden_outs = add_bias(hidden_acts, self.bias)             # (N, H+1)
        outputs = hidden_outs @ self.weights2                      # (N, 1)
        return hidden_outs, outputs

    @staticmethod
    def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Map linear outputs through logistic to get probabilities.
        Returns shape (N,)
        """
        Xb = add_bias(X, self.bias)
        _, outputs = self.forward(Xb)     # (N, 1)
        probs = logistic(outputs[:, 0])
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        p = self.predict_probabilities(X)
        return (p > threshold).astype(int)

    def fit(
        self,
        X_train: np.ndarray,
        t_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        t_val: Optional[np.ndarray] = None,
        lr: float = 1e-3,
        epochs: int = 2000,
        tol: float = 0.0,
        n_epochs_no_update: int = 2,
        random_state: int = 0
    ):
        rng = np.random.RandomState(random_state)
        self.losses = {"train": [], "val": []}
        self.accs = {"train": [], "val": []}
        self.n_epochs = 0

        T_train = t_train.reshape(-1, 1)
        dim_in = X_train.shape[1]
        dim_out = 1
        H = self.dim_hidden

        # Init weights with fan-in scaling
        self.weights1 = (rng.rand(dim_in + 1, H) * 2 - 1) / np.sqrt(dim_in + 1)
        self.weights2 = (rng.rand(H + 1, dim_out) * 2 - 1) / np.sqrt(H + 1)

        Xb_train = add_bias(X_train, self.bias)

        best_loss = np.inf
        epochs_no_improve = 0

        for e in range(epochs):
            self.n_epochs += 1

            # Forward
            hidden_outs, outputs = self.forward(Xb_train)  # (N, H+1), (N, 1)

            # Deltas
            out_deltas = (outputs - T_train)               # (N, 1)
            hiddenout_diffs = out_deltas @ self.weights2.T # (N, H+1)
            hiddenact_deltas = hiddenout_diffs[:, 1:] * self.activ_diff(hidden_outs[:, 1:])  # (N, H)

            # Loss/acc (train)
            train_loss = self.mse(outputs, T_train)
            self.losses["train"].append(train_loss)
            self.accs["train"].append(accuracy(self.predict(X_train), t_train))

            # Weight updates (batch GD)
            self.weights2 -= lr * (hidden_outs.T @ out_deltas)
            self.weights1 -= lr * (Xb_train.T @ hiddenact_deltas)

            # Validation tracking
            if X_val is not None and t_val is not None:
                Xb_val = add_bias(X_val, self.bias)
                _, val_outputs = self.forward(Xb_val)
                val_loss = self.mse(val_outputs, t_val.reshape(-1, 1))
                self.losses["val"].append(val_loss)
                self.accs["val"].append(accuracy(self.predict(X_val), t_val))

            # Early stopping based on training loss
            if train_loss < best_loss - tol:
                best_loss = train_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= n_epochs_no_update:
                    break

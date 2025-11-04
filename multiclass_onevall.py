import numpy as np
from utils import accuracy
from logistic import NumpyLogReg

class MultiClassifier:
    """
    One-vs-rest multi-class classifier using binary logistic regression as base.
    """
    def __init__(self):
        self.labels = None
        self.cls = []

    def fit(self, X_train: np.ndarray, t_multi_train: np.ndarray,
            eta: float = 0.1, epochs: int = 100):
        self.labels = np.unique(t_multi_train)
        self.cls = []
        for class_label in self.labels:
            t2 = (t_multi_train == class_label).astype(int)
            lr_cl = NumpyLogReg()
            lr_cl.fit(X_train, t2, eta=eta, epochs=epochs)
            self.cls.append(lr_cl)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Returns matrix (N, n_classes) of probabilities from each OVR classifier.
        Rows are normalized to sum to 1 for interpretability.
        """
        prob_matrix = np.zeros((X.shape[0], len(self.labels)))
        for i, clf in enumerate(self.cls):
            prob_matrix[:, i] = clf.predict_probabilities(X)
        denom = prob_matrix.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        return prob_matrix / denom

    def predict(self, X: np.ndarray) -> np.ndarray:
        prob_matrix = self.predict_probabilities(X)
        class_indices = np.argmax(prob_matrix, axis=1)
        return self.labels[class_indices]

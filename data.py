import numpy as np
from sklearn.datasets import make_blobs

def make_data(random_state: int = 424242):
    """
    Returns:
      X_train, X_val, X_test,
      t_multi_train, t_multi_val, t_multi_test,
      t2_train, t2_val, t2_test
    """
    # Generate synthetic dataset: 10 classes, 500 instances per class (total 5000)
    X, t_multi = make_blobs(
        n_samples=[500] * 10,
        n_features=2,
        random_state=random_state,
        cluster_std=[1.0, 2.0, 1.5, 1.0, 1.5, 2.0, 1.5, 1.2, 1.8, 2.0],
        centers=None
    )

    # Shuffle and split: 60% train, 20% val, 20% test
    indices = np.arange(X.shape[0])
    rng = np.random.RandomState(random_state)
    rng.shuffle(indices)

    n_train = int(0.6 * len(indices))
    n_val = int(0.2 * len(indices))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train = X[train_idx, :]
    X_val = X[val_idx, :]
    X_test = X[test_idx, :]

    t_multi_train = t_multi[train_idx]
    t_multi_val = t_multi[val_idx]
    t_multi_test = t_multi[test_idx]

    # Binary labels: 0-5 -> 0, 6-9 -> 1
    t2_train = (t_multi_train >= 6).astype(int)
    t2_val = (t_multi_val >= 6).astype(int)
    t2_test = (t_multi_test >= 6).astype(int)

    return (
        X_train, X_val, X_test,
        t_multi_train, t_multi_val, t_multi_test,
        t2_train, t2_val, t2_test
    )

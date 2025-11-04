import numpy as np
import matplotlib.pyplot as plt

from data import make_data
from utils import plot_decision_regions, standard_scaler, accuracy
from mlp_binary import MLPBinaryLinRegClass

def plot_losses_accuracies(model, title_prefix="MLP (Binary)"):
    epochs = model.n_epochs
    x = np.arange(1, epochs + 1)

    plt.figure(figsize=(12, 5))
    # Losses
    plt.subplot(1, 2, 1)
    plt.plot(x, model.losses["train"], "b-", label="Train Loss")
    if model.losses["val"]:
        plt.plot(x, model.losses["val"], "r-", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix}: Loss vs Epochs")
    plt.legend()
    plt.grid(True)

    # Accuracies
    plt.subplot(1, 2, 2)
    plt.plot(x, model.accs["train"], "b-", label="Train Acc")
    if model.accs["val"]:
        plt.plot(x, model.accs["val"], "r-", label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix}: Accuracy vs Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def hyperparam_search(X_train, t_train, X_val, t_val,
                      dim_list=(6, 7, 8, 9, 10),
                      eta_list=(5e-4, 1e-4, 5e-5),
                      tol_list=(1e-3, 1e-4, 1e-5, 1e-6)):
    best = {"acc": -np.inf}
    results = []
    for dim in dim_list:
        for eta in eta_list:
            for tol in tol_list:
                model = MLPBinaryLinRegClass(dim_hidden=dim)
                model.fit(X_train, t_train, X_val=X_val, t_val=t_val,
                          lr=eta, epochs=20000, tol=tol, n_epochs_no_update=2)
                acc = accuracy(model.predict(X_val), t_val)
                results.append((dim, eta, tol, acc))
                if acc > best["acc"]:
                    best = {"dim": dim, "eta": eta, "tol": tol, "acc": acc, "model": model}
    return best, results

def main():
    (
        X_train, X_val, X_test,
        t_multi_train, t_multi_val, t_multi_test,
        t2_train, t2_val, t2_test
    ) = make_data()

    # Optional: scale inputs to improve convergence
    X_train_s, X_val_s = standard_scaler(X_train, X_val)

    # Hyperparameter search
    best, results = hyperparam_search(X_train_s, t2_train, X_val_s, t2_val)
    print(f"[MLP] Best: dim={best['dim']}, lr={best['eta']}, tol={best['tol']}, val_acc={best['acc']:.3f}")

    # Plot loss and accuracy curves for the best model
    plot_losses_accuracies(best["model"], title_prefix="MLP (Binary, Best Model)")

    # Decision regions for best MLP
    plot_decision_regions(X_val_s, t2_val, best["model"], size=(8, 6), cmap="tab10")

    # Non-determinism: run best setting 3 times and report mean/std
    accs = []
    for seed in (0, 1, 2):
        m = MLPBinaryLinRegClass(dim_hidden=best["dim"])
        m.fit(X_train_s, t2_train, X_val=X_val_s, t_val=t2_val,
              lr=best["eta"], epochs=20000, tol=best["tol"], n_epochs_no_update=2, random_state=seed)
        accs.append(accuracy(m.predict(X_val_s), t2_val))
    print(f"[MLP] Validation accuracy over 3 runs: mean={np.mean(accs):.3f}, std={np.std(accs):.3f}")

if __name__ == "__main__":
    main()

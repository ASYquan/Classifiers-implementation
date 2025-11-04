
plot_lin_log.py

import numpy as np
import matplotlib.pyplot as plt

from data import make_data
from utils import plot_decision_regions, standard_scaler, accuracy
from linreg import NumpyLinRegClass, grid_search_linreg
from logistic import NumpyLogReg, grid_search_logreg

def plot_linreg_contour(results, title="Linear Regression: Accuracy by LR and Epochs"):
    # results: list of (lr, epochs, acc)
    lrs = sorted(set(r[0] for r in results))
    epochs = sorted(set(r[1] for r in results))

    lr_to_idx = {lr: i for i, lr in enumerate(lrs)}
    ep_to_idx = {ep: j for j, ep in enumerate(epochs)}

    acc_grid = np.zeros((len(epochs), len(lrs)))
    for lr, ep, acc in results:
        acc_grid[ep_to_idx[ep], lr_to_idx[lr]] = acc

    LR, EP = np.meshgrid(lrs, epochs)

    plt.figure(figsize=(12, 7))
    cp = plt.contourf(EP, LR, acc_grid, levels=20, cmap="viridis")
    plt.colorbar(cp, label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_losses_accuracies(model, title_prefix="Logistic Regression"):
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

def main():
    (
        X_train, X_val, X_test,
        t_multi_train, t_multi_val, t_multi_test,
        t2_train, t2_val, t2_test
    ) = make_data()

    # Optional: scale train/val for experiments
    X_train_s, X_val_s = standard_scaler(X_train, X_val)

    # Linear regression grid search (on scaled data recommended)
    lrs = [0.005, 0.01, 0.015, 0.02, 0.05]
    epochs_list = [50, 75, 100, 115, 130, 200]
    best_lin, lin_results = grid_search_linreg(X_train_s, t2_train, X_val_s, t2_val, lrs, epochs_list)
    best_lr, best_epochs, best_acc = best_lin
    print(f"[Linear] Best: lr={best_lr}, epochs={best_epochs}, val_acc={best_acc:.3f}")
    plot_linreg_contour(lin_results, title="Linear Regression (Scaled): Accuracy by LR and Epochs")

    # Train best linear and plot decision regions on validation
    lin_best = NumpyLinRegClass()
    lin_best.fit(X_train_s, t2_train, lr=best_lr, epochs=best_epochs)
    print(f"[Linear] Val accuracy (best): {accuracy(lin_best.predict(X_val_s), t2_val):.3f}")
    plot_decision_regions(X_val_s, t2_val, lin_best, size=(8, 6), cmap="tab10")

    # Logistic regression grid search
    best_log, log_results = grid_search_logreg(
        X_train, t2_train, X_val, t2_val,
        eta_list=(1, 0.1, 0.01, 0.001, 0.0001),
        epochs_list=(50, 100, 300, 500, 1000),
        tol_list=(1e-3, 1e-4, 1e-5)
    )
    be_epochs, be_eta, be_tol, be_acc, be_model = best_log
    print(f"[Logistic] Best: eta={be_eta}, epochs={be_epochs}, tol={be_tol}, val_acc={be_acc:.3f}")

    # Plot training/validation losses and accuracies for best logistic model
    plot_losses_accuracies(be_model, title_prefix="Logistic Regression (Best Model)")

    # Decision regions for logistic (validation)
    plot_decision_regions(X_val, t2_val, be_model, size=(8, 6), cmap="tab10")

if __name__ == "__main__":
    main()

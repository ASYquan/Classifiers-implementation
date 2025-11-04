# IN3050/IN4050 Assignment 2 — Supervised Learning (Code-Only Repo)

This repository contains a cleaned, runnable Python implementation extracted from the assignment notebook/PDF. It organizes code by topic and adds scripts to reproduce key plots and metrics.

## Overview

Implemented models:
- Linear regression classifier using MSE loss
- Binary logistic regression with early stopping, tracked losses/accuracies
- One-vs-rest multi-class logistic regression
- Simple MLP for binary classification (one hidden layer)

Utility modules:
- Synthetic dataset generator and splits
- Bias handling, scaling, accuracy
- Decision region plotting

## Structure

- `data.py` — dataset generation and splitting (multi-class and binary labels)
- `utils.py` — helper utilities: bias handling, accuracy, scaling, decision region plotting
- `linreg.py` — linear regression classifier and a simple grid search
- `logistic.py` — binary logistic regression with early stopping, loss/acc tracking, and grid search
- `multiclass_ovr.py` — one-vs-rest multi-class classifier built from binary logistic regression
- `mlp_binary.py` — simple MLP for binary classification with early stopping and tracking
- `final_eval.py` — trains best models found and reports accuracy, precision, recall across splits
- `plot_lin_log.py` — reproduces tuning and plots for linear/logistic models
- `plot_mlp.py` — reproduces tuning and plots for the MLP, including loss/accuracy curves and decision regions

## Setup

1) Python 3.9+ recommended
2) Install dependencies:
```bash
pip install numpy matplotlib scikit-learn

import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

from data import make_data
from linreg import NumpyLinRegClass
from logistic import NumpyLogReg
from mlp_binary import MLPBinaryLinRegClass

def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1)
    return acc, prec, rec

def main():
    (
        X_train, X_val, X_test,
        t_multi_train, t_multi_val, t_multi_test,
        t2_train, t2_val, t2_test
    ) = make_data()

    # Instantiate with best params found in the assignment narrative
    linear_reg_model = NumpyLinRegClass()
    logistic_reg_model = NumpyLogReg()
    multi_layer_model = MLPBinaryLinRegClass(dim_hidden=9)

    # Fit
    linear_reg_model.fit(X_train, t2_train, lr=0.015, epochs=115)
    logistic_reg_model.fit(X_train, t2_train, eta=0.1, epochs=1000, tol=1e-5)
    multi_layer_model.fit(X_train, t2_train, lr=1e-4, epochs=20000, tol=1e-6)

    # Predictions
    train_pred_lin = linear_reg_model.predict(X_train)
    val_pred_lin = linear_reg_model.predict(X_val)
    test_pred_lin = linear_reg_model.predict(X_test)

    train_pred_log = logistic_reg_model.predict(X_train)
    val_pred_log = logistic_reg_model.predict(X_val)
    test_pred_log = logistic_reg_model.predict(X_test)

    train_pred_mlp = multi_layer_model.predict(X_train)
    val_pred_mlp = multi_layer_model.predict(X_val)
    test_pred_mlp = multi_layer_model.predict(X_test)

    results = {
        "Linear Regression": {
            "Train": calculate_metrics(t2_train, train_pred_lin),
            "Validation": calculate_metrics(t2_val, val_pred_lin),
            "Test": calculate_metrics(t2_test, test_pred_lin),
        },
        "Logistic Regression": {
            "Train": calculate_metrics(t2_train, train_pred_log),
            "Validation": calculate_metrics(t2_val, val_pred_log),
            "Test": calculate_metrics(t2_test, test_pred_log),
        },
        "Multi-Layer Network": {
            "Train": calculate_metrics(t2_train, train_pred_mlp),
            "Validation": calculate_metrics(t2_val, val_pred_mlp),
            "Test": calculate_metrics(t2_test, test_pred_mlp),
        },
    }

    # Pretty print
    print(f"{'Model':<20} {'Split':<12} {'Acc':>6} {'Prec':>6} {'Rec':>6}")
    for model_name, splits in results.items():
        for split_name, (acc, prec, rec) in splits.items():
            print(f"{model_name:<20} {split_name:<12} {acc:6.2f} {prec:6.2f} {rec:6.2f}")

if __name__ == "__main__":
    main()

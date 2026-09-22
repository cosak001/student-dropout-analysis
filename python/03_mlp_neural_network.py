"""
Student dropout prediction, step 3: the MLP neural network.

The original coursework used an SPSS MLP alongside the logistic regression.
This reproduces it in scikit-learn and, more importantly, asks whether the
extra complexity actually buys anything.

Two things differ from script 02:

  1. Scaling is required. HS_GPA runs about 0 to 4 while Unit_Enrolled runs
     about 3 to 18. Logistic regression is unbothered by that, but a neural
     network is not, because the raw size of an input affects how much it
     moves the weights. StandardScaler puts every predictor on the same
     footing.

  2. MLPClassifier has no class_weight option, so the imbalance fix has to
     come from threshold tuning alone.

Scaler and model are wrapped in a Pipeline so the scaler is fit on training
data only. Scaling the whole dataset first would leak test-set information
into training, which quietly inflates results.

Run it from the repo root with:
    python python/03_mlp_neural_network.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "student_retention.csv"
IMAGES = REPO_ROOT / "images"

RANDOM_STATE = 42
RECALL_TARGET = 0.60

FEATURES = [
    "HS_GPA",
    "Unit_Enrolled",
    "Age",
    "Seminar_D",
    "Gender_Dummy",
    "Registration_Status_Dummy",
    "Commuter_Dummy",
]
TARGET = "Dropped_Out_Dummy"


def load_data():
    df = pd.read_csv(DATA)
    df.columns = [c.strip() for c in df.columns]
    return df


def make_mlp(seed=RANDOM_STATE):
    """One hidden layer of 5 neurons, matching the scale of the SPSS model.

    With 600 rows and 7 predictors, a big network would memorize the training
    data rather than learn from it. Small is the right choice here, not a
    limitation.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(5,),
            max_iter=3000,
            random_state=seed,
        )),
    ])


def make_logistic():
    return LogisticRegression(C=1e9, max_iter=5000)


def report(title, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = (tp + tn) / len(y_true)

    print(f"--- {title} ---")
    print(f"  Dropouts caught:        {tp} of {tp + fn}   (recall {recall:.0%})")
    print(f"  False alarms:           {fp} of {fp + tn} students who stayed")
    print(f"  Of those flagged,       {precision:.0%} actually dropped out")
    print(f"  Overall accuracy:       {accuracy:.1%}")
    print()


def choose_threshold(model, X_train, y_train):
    """Same approach as script 02. Cutoff comes from training data only."""
    cv_probs = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        method="predict_proba",
    )[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_train, cv_probs)
    precisions, recalls = precisions[:-1], recalls[:-1]

    eligible = recalls >= RECALL_TARGET
    if not eligible.any():
        return 0.50
    return float(thresholds[np.argmax(np.where(eligible, precisions, -1))])


def stability_check(X, y):
    """Train the same MLP with different random seeds.

    Neural networks start from random weights, so two runs on identical data
    give slightly different models. Logistic regression has one solution and
    always finds it. If the MLP's spread across seeds is wider than the gap
    between the two models, then any apparent MLP advantage is noise.
    """
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for seed in range(5):
        score = cross_val_score(make_mlp(seed), X, y, cv=cv, scoring="roc_auc").mean()
        aucs.append(score)
        print(f"  seed {seed}: AUC {score:.3f}")
    print(f"  spread across seeds: {min(aucs):.3f} to {max(aucs):.3f}\n")
    return aucs


def main():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    # ---- Head to head on ranking quality ---------------------------------
    print("Cross-validated AUC on the full dataset:")
    log_cv = cross_val_score(make_logistic(), X, y, cv=cv, scoring="roc_auc")
    mlp_cv = cross_val_score(make_mlp(), X, y, cv=cv, scoring="roc_auc")
    print(f"  Logistic regression: {log_cv.mean():.3f} (+/- {log_cv.std():.3f})")
    print(f"  MLP neural network:  {mlp_cv.mean():.3f} (+/- {mlp_cv.std():.3f})\n")

    # ---- How much of that is just the random seed? -----------------------
    print("MLP stability across five random seeds:")
    stability_check(X, y)

    # ---- Held-out performance --------------------------------------------
    mlp = make_mlp().fit(X_train, y_train)
    mlp_probs = mlp.predict_proba(X_test)[:, 1]

    logistic = make_logistic().fit(X_train, y_train)
    log_probs = logistic.predict_proba(X_test)[:, 1]

    print(f"Held-out AUC, logistic: {roc_auc_score(y_test, log_probs):.3f}")
    print(f"Held-out AUC, MLP:      {roc_auc_score(y_test, mlp_probs):.3f}\n")

    report("MLP, default cutoff 0.50", y_test, mlp.predict(X_test))

    threshold = choose_threshold(make_mlp(), X_train, y_train)
    print(f"MLP tuned cutoff: {threshold:.3f}\n")
    report(f"MLP, tuned cutoff {threshold:.3f}", y_test, (mlp_probs >= threshold).astype(int))

    # ---- One figure, both models -----------------------------------------
    IMAGES.mkdir(exist_ok=True)
    plt.figure(figsize=(6, 5))
    for probs, name in [(log_probs, "Logistic regression"), (mlp_probs, "MLP neural network")]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, probs):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random guessing")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Logistic regression vs MLP, held-out test set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(IMAGES / "model_comparison_roc.png", dpi=150)
    plt.close()
    print(f"Saved comparison curve to {IMAGES / 'model_comparison_roc.png'}")


if __name__ == "__main__":
    main()

"""
Student dropout prediction, step 2: making the model actually useful.

Script 01 showed the problem. The model was 81% accurate but caught only 3 of
30 real dropouts, because a baseline that guesses "stayed" every time is
already 80% accurate. A retention office cannot act on a model that never
flags anyone.

This script fixes that in two steps:

  1. class_weight="balanced" tells the model that missing a dropout is as
     costly as a false alarm, instead of letting the majority class dominate.

  2. Threshold tuning. predict() uses a hard cutoff of 0.50 by default, which
     is arbitrary. We test many cutoffs and pick one that catches a useful
     share of dropouts at an acceptable false alarm rate.

The threshold is chosen using cross-validation on the TRAINING data only.
Picking it by looking at test results would leak the test set into the model
and make the reported numbers optimistic.

Run it from the repo root with:
    python python/02_class_weights_and_threshold.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to file instead of opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

# Paths are built from this file's location, so the script runs correctly no
# matter which folder your terminal is sitting in.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "student_retention.csv"
IMAGES = REPO_ROOT / "images"

RANDOM_STATE = 42

# How many real dropouts we want to catch. 0.60 means the model should flag
# about 60 of every 100 students who go on to leave. This is a policy choice,
# not a statistical one, and it belongs to whoever runs the intervention.
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


def make_model(balanced):
    """Same logistic regression as script 01, with weighting optional.

    C=1e9 effectively turns off scikit-learn's default L2 penalty so the
    coefficients match the original SPSS output.
    """
    return LogisticRegression(
        C=1e9,
        max_iter=5000,
        class_weight="balanced" if balanced else None,
    )


def report(title, y_true, y_pred):
    """Print a confusion matrix in words rather than a bare 2x2 grid."""
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
    """Pick a cutoff using cross-validation inside the training set.

    cross_val_predict gives each training row a predicted probability from a
    model that never saw that row, so these probabilities are honest even
    though they come from training data.
    """
    cv_probs = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        method="predict_proba",
    )[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_train, cv_probs)

    # precision_recall_curve returns one more precision/recall than thresholds,
    # so trim the last pair to line the arrays up.
    precisions, recalls = precisions[:-1], recalls[:-1]

    print("Threshold options from cross-validation on the training set:")
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        pred = (cv_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, pred).ravel()
        r = tp / (tp + fn) if (tp + fn) else 0.0
        p = tp / (tp + fp) if (tp + fp) else 0.0
        print(f"  cutoff {t:.2f}   catches {r:5.0%} of dropouts   {p:5.0%} of flags are correct")
    print()

    # Of every cutoff that reaches the recall target, take the one with the
    # best precision, so we hit the goal with the fewest false alarms.
    eligible = recalls >= RECALL_TARGET
    if not eligible.any():
        print(f"No cutoff reached {RECALL_TARGET:.0%} recall. Falling back to 0.50.")
        return 0.50

    best = np.argmax(np.where(eligible, precisions, -1))
    return float(thresholds[best])


def save_curves(y_test, probs):
    IMAGES.mkdir(exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Logistic regression (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random guessing (AUC = 0.500)")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve, held-out test set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(IMAGES / "roc_curve.png", dpi=150)
    plt.close()

    precisions, recalls, _ = precision_recall_curve(y_test, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, label="Logistic regression")
    plt.axhline(y_test.mean(), ls="--", color="gray", label="Base dropout rate")
    plt.xlabel("Recall, share of real dropouts caught")
    plt.ylabel("Precision, share of flags that are correct")
    plt.title("Precision-recall curve, held-out test set")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(IMAGES / "precision_recall_curve.png", dpi=150)
    plt.close()

    print(f"Saved both curves to {IMAGES}")


def main():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    # ---- Where script 01 left off ---------------------------------------
    plain = make_model(balanced=False).fit(X_train, y_train)
    report("Script 01 model, cutoff 0.50", y_test, plain.predict(X_test))

    # ---- Step 1, balance the classes ------------------------------------
    balanced = make_model(balanced=True).fit(X_train, y_train)
    report("Balanced class weights, cutoff 0.50", y_test, balanced.predict(X_test))

    # ---- Step 2, tune the cutoff on training data only -------------------
    threshold = choose_threshold(make_model(balanced=False), X_train, y_train)
    print(f"Chosen cutoff: {threshold:.3f}\n")

    probs = plain.predict_proba(X_test)[:, 1]
    report(f"Script 01 model, tuned cutoff {threshold:.3f}", y_test, (probs >= threshold).astype(int))

    # ---- AUC is unchanged by any of this ---------------------------------
    # Class weights and thresholds change where the line is drawn, not how
    # well the model ranks students. AUC measures the ranking, so it stays
    # roughly the same. That is the point of reporting it.
    print(f"Held-out ROC AUC: {roc_auc_score(y_test, probs):.3f}")
    print(f"Balanced model AUC: {roc_auc_score(y_test, balanced.predict_proba(X_test)[:, 1]):.3f}\n")

    save_curves(y_test, probs)


if __name__ == "__main__":
    main()

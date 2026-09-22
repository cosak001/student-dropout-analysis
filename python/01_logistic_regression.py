"""
Student dropout prediction, step 1: logistic regression in scikit-learn.

This reproduces the SPSS logistic regression from the original coursework and
then evaluates it honestly, on data the model has never seen.

Run it with:
    python 01_logistic_regression.py
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

DATA = "data/student_retention.csv"
RANDOM_STATE = 42

# The seven predictors, using the dummy columns that already exist in the file.
# Seminar_D, Gender_Dummy, Registration_Status_Dummy and Commuter_Dummy are all
# already 0/1, so no extra encoding is needed here.
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


def load_data(path=DATA):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]  # one column has a trailing space
    return df


def make_model():
    """Logistic regression with regularization effectively turned off.

    SPSS does not regularize. Scikit-learn does by default, so out of the box
    the coefficients will NOT match SPSS. Setting C very high shrinks the
    penalty to almost nothing, which reproduces the SPSS fit. C is the inverse
    of regularization strength, so a big C means a small penalty.
    """
    return LogisticRegression(C=1e9, max_iter=5000)


def main():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    print(f"Rows: {len(df)}   Dropouts: {y.sum()} ({y.mean():.1%})")

    # ---- Baseline -------------------------------------------------------
    baseline = DummyClassifier(strategy="most_frequent").fit(X, y)
    print(f"Baseline accuracy (guess the majority class): {baseline.score(X, y):.1%}\n")

    # ---- Train / test split ---------------------------------------------
    # stratify=y keeps the same dropout rate in both halves, which matters a
    # lot when only 20% of rows are positive.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    model = make_model().fit(X_train, y_train)

    # ---- Coefficients and odds ratios -----------------------------------
    print("Coefficients (odds ratio in brackets):")
    for name, coef in zip(FEATURES, model.coef_[0]):
        print(f"  {name:28s} {coef:8.4f}  [{np.exp(coef):6.3f}]")
    print(f"  {'intercept':28s} {model.intercept_[0]:8.4f}\n")

    # ---- Evaluation on held-out data ------------------------------------
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"Held-out accuracy: {accuracy_score(y_test, predictions):.1%}")
    print(f"Held-out ROC AUC:  {roc_auc_score(y_test, probabilities):.3f}")

    cv_auc = cross_val_score(
        make_model(),
        X,
        y,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
    )
    print(f"5-fold CV ROC AUC: {cv_auc.mean():.3f} (+/- {cv_auc.std():.3f})\n")

    print(classification_report(y_test, predictions, target_names=["stayed", "dropped"]))


if __name__ == "__main__":
    main()

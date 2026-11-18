"""Classification of "high pollution" readings, with and without data leakage.

Label: 1 if `value` is above its 65th percentile. If `value` is left among the
features, the model is handed the answer (data leakage). Four experiments isolate
each change:

cpp_replica_leaky     Replicates the C++ project (with `value` as a feature).
no_leakage_cpp_setup  Same as 1, removing only `value`.
no_leakage_clean      Removes `value` and also: implausible readings dropped,
                      shuffled stratified split, scaler fitted on train only,
                      one-hot encoding of pollutant and country.
per_pollutant_label   Like 3, but the label is relative to each (pollutant, unit).

`robustness_checks` then stresses experiments 3 and 4 with 5-fold cross-validation:
it holds out whole stations or whole countries, and compares against a baseline that
only sees pollutant + unit.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .analysis import cpp_style_features
from .data import clean_for_modelling

QUANTILE = 0.65
NUMERIC = ["lat", "lon"]
CATEGORICAL = ["pollutant", "country"]  # `value` is deliberately NOT here
TEST_SIZE = 0.2
N_FOLDS = 5


def _threshold(values: pd.Series) -> float:
    # Same rule as the C++ code: element int(n * 0.65) of the sorted vector
    return np.sort(values.to_numpy())[int(len(values) * QUANTILE)]


def global_label(df: pd.DataFrame) -> pd.Series:
    return (df["value"] > _threshold(df["value"])).astype(int)


def per_pollutant_label(df: pd.DataFrame) -> pd.Series:
    thr = df.groupby(["pollutant", "unit"], dropna=False)["value"].transform(_threshold)
    return (df["value"] > thr).astype(int)


def score(y_true, y_pred, y_score=None) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    out = {"precision": precision, "recall": recall, "f1": f1}
    if y_score is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    return out


# --- Experiments 1 and 2: the original C++ setup -----------------------------------

def _cpp_setup(df: pd.DataFrame, drop_value: bool) -> dict:
    X = cpp_style_features(df)
    y = global_label(df)
    if drop_value:
        X = X.drop(columns="value")
    n_train = int(len(X) * 0.8)  # sequential split without shuffling, as in C++
    X_tr, y_tr, X_te, y_te = X[:n_train], y[:n_train], X[n_train:], y[n_train:]
    classifiers = {
        # Per-sample SGD, lr=0.1, 1000 epochs, no shuffling: equivalent to the C++ train()
        "logreg_sgd_as_cpp": SGDClassifier(
            loss="log_loss", penalty=None, learning_rate="constant", eta0=0.1,
            max_iter=1000, tol=None, shuffle=False,
        ),
        # Optimum of the unregularised logistic regression (which SGD never reaches)
        "logreg_lbfgs": LogisticRegression(C=np.inf, max_iter=5000),
    }
    models = {}
    for name, clf in classifiers.items():
        clf.fit(X_tr, y_tr)
        models[name] = score(y_te, clf.predict(X_te), clf.predict_proba(X_te)[:, 1])
    return {
        "features": list(X.columns),
        "positive_rate": float(y.mean()),
        "n_train": n_train,
        "n_test": len(X_te),
        "models": models,
    }


# --- Experiments 3 and 4 and the robustness checks ----------------------------------

def station_id(df: pd.DataFrame) -> np.ndarray:
    """A station is identified by its coordinates (location names are not unique)."""
    return df.groupby(["lat", "lon"]).ngroup().to_numpy()


def _preprocessor(numeric, categorical) -> ColumnTransformer:
    steps = []
    if numeric:
        steps.append(("num", StandardScaler(), numeric))
    if categorical:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
    return ColumnTransformer(steps)


def _classifiers(seed: int) -> dict:
    return {
        "logreg": LogisticRegression(max_iter=1000),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
    }


def _clean_setup(df, y, seed, numeric=NUMERIC, categorical=CATEGORICAL) -> dict:
    """Shuffled, stratified 80/20 hold-out split."""
    X = df[numeric + categorical]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=seed)
    models = {}
    for name, clf in _classifiers(seed).items():
        # The pipeline is fitted on the training rows only, so scaling never sees test data
        pipe = make_pipeline(_preprocessor(numeric, categorical), clf).fit(X_tr, y_tr)
        models[name] = score(y_te, pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1])
    return {
        "features": numeric + categorical,
        "positive_rate": float(y.mean()),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "models": models,
    }


def _cross_validate(df, y, seed, how="random", numeric=NUMERIC, categorical=CATEGORICAL) -> dict:
    """5-fold CV; metrics are computed on the pooled out-of-fold predictions.

    `how` is "random" (stratified folds), "by_station" or "by_country" (no station or
    country appears in both the training and the test folds).
    """
    X = df[numeric + categorical]
    if how == "random":
        cv, groups = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed), None
    else:
        cv = GroupKFold(N_FOLDS)
        groups = station_id(df) if how == "by_station" else df["country"].to_numpy()
    models = {}
    for name, clf in _classifiers(seed).items():
        pipe = make_pipeline(_preprocessor(numeric, categorical), clf)
        proba = cross_val_predict(pipe, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
        models[name] = score(y, (proba > 0.5).astype(int), proba)
    return {"features": numeric + categorical, "cv": f"{N_FOLDS}-fold, {how}", "models": models}


def robustness_checks(clean: pd.DataFrame, seed: int = 42) -> dict:
    """How much of the clean score survives harder evaluations, for both labels."""
    out = {}
    for label_name, label_fn in [("global", global_label), ("per_pollutant", per_pollutant_label)]:
        y = label_fn(clean)
        out[label_name] = {
            "cv_random": _cross_validate(clean, y, seed),
            "baseline_pollutant_unit_only": _cross_validate(
                clean, y, seed, numeric=[], categorical=["pollutant", "unit"]
            ),
            "cv_by_station": _cross_validate(clean, y, seed, how="by_station"),
            "cv_by_country": _cross_validate(clean, y, seed, how="by_country"),
        }
    return out


def run_experiments(df: pd.DataFrame, seed: int = 42) -> dict:
    """`df` is the output of `load_data`. Experiments 3 and 4 clean it further."""
    clean = clean_for_modelling(df)
    return {
        "cpp_replica_leaky": _cpp_setup(df, drop_value=False),
        "no_leakage_cpp_setup": _cpp_setup(df, drop_value=True),
        "no_leakage_clean": _clean_setup(clean, global_label(clean), seed),
        "per_pollutant_label": _clean_setup(clean, per_pollutant_label(clean), seed),
    }

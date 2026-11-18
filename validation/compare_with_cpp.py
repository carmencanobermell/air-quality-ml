"""Compare the Python rebuild (pandas + scikit-learn) with the original C++ implementation.

Usage:  python validation/compare_with_cpp.py
Exits with code 1 if any check fails. See build_cpp_reference.py for how
validation/cpp_reference.json was generated.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from airquality.analysis import association_rules, correlations, cpp_style_features  # noqa: E402
from airquality.classify import global_label  # noqa: E402
from airquality.data import load_data  # noqa: E402
from build_cpp_reference import FILTERED_CSV, inertia, write_filtered_csv  # noqa: E402

warnings.filterwarnings("ignore")
ref = json.loads((HERE / "cpp_reference.json").read_text())
checks = []  # (name, C++ value, Python value, ok)


def check(name, cpp, py, atol):
    checks.append((name, cpp, py, abs(cpp - py) <= atol))


if not FILTERED_CSV.exists():
    write_filtered_csv(FILTERED_CSV)
df = load_data(FILTERED_CSV)
X = cpp_style_features(df)

# --- Loading ---
check("rows loaded", ref["n_rows"], len(df), 0)

# --- Correlations (C++ prints 6 significant digits) ---
corr = correlations(X)
check("corr lat-value", ref["correlation"]["lat_value"], corr.loc["lat", "value"], 5e-6)
check("corr lon-value", ref["correlation"]["lon_value"], corr.loc["lon", "value"], 5e-6)
check("corr lat-lon", ref["correlation"]["lat_lon"], corr.loc["lat", "lon"], 5e-6)

# --- Rules: the support of every pair must match ---
LEVELS = {"bajo": "low", "baja": "low", "medio": "medium", "media": "medium", "alto": "high", "alta": "high"}


def cpp_name(item):  # C++ names: lat_alta -> lat=high ; val_bajo -> value=low
    kind, level = item.split("_")
    return {"lat": "lat", "lon": "lon", "val": "value"}[kind] + "=" + LEVELS[level]

rules = association_rules(X, min_support=0.1).set_index(["antecedent", "consequent"])["support"]
for pair, support in ref["rules_support"].items():
    a, b = (cpp_name(i) for i in pair.split(" & "))
    check(f"support {a} & {b}", support, rules[(a, b)], 5e-6)
check("number of rules (support>=0.1)", len(ref["rules_support"]), len(rules), 0)

# --- Logistic regression: same algorithm (per-sample SGD, lr 0.1, 1000 epochs) ---
y = global_label(df)
n = int(len(X) * 0.8)
check("positives", ref["positives"], int(y.sum()), 0)
sgd = SGDClassifier(
    loss="log_loss", penalty=None, learning_rate="constant", eta0=0.1,
    max_iter=1000, tol=None, shuffle=False,
).fit(X[:n], y[:n])
for i, (w_cpp, w_py) in enumerate(zip(ref["weights"], np.r_[sgd.intercept_, sgd.coef_[0]])):
    check(f"weight w{i}", w_cpp, w_py, 1e-4 * max(1, abs(w_cpp)))
precision, recall, f1, _ = precision_recall_fscore_support(y[n:], sgd.predict(X[n:]), average="binary")
check("precision", ref["metrics"]["precision"], precision, 5e-6)
check("recall", ref["metrics"]["recall"], recall, 5e-6)
check("F1", ref["metrics"]["f1"], f1, 5e-6)

# --- K-Means: initialisation is random, so the inertia distribution is compared ---
py_inertias = [
    inertia(X.to_numpy(), KMeans(5, init="random", n_init=1, max_iter=100, random_state=s).fit(X).labels_)
    for s in range(10)
]
cpp_in = np.array(ref["kmeans_inertias"])
py_in = np.array(py_inertias)
check("K-Means min inertia (10 runs)", cpp_in.min(), py_in.min(), 0.01 * cpp_in.min())
check("K-Means median inertia", np.median(cpp_in), np.median(py_in), 0.03 * np.median(cpp_in))

# --- Report ---
pd.options.display.float_format = "{:.6g}".format
report = pd.DataFrame(checks, columns=["check", "C++", "Python", "ok"])
report["ok"] = report["ok"].map({True: "OK", False: "FAIL"})
print(report.to_string(index=False))
failed = [c for c in checks if not c[3]]
print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
sys.exit(1 if failed else 0)

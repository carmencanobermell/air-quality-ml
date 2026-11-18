"""Exploratory part reproduced from the original C++ project: K-Means, correlations
and association rules.

These functions are kept so the Python rebuild can be validated one to one against
the C++ code (see validation/). They have little analytical value on this data:
the feature matrix uses ordinal indices for categorical columns and min-max scales
`value` over a column with extreme outliers, so 99.98% of rows fall in the lowest
`value` bin. The README explains this; the classification experiments do not use it.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def cpp_style_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix exactly as the C++ loader built it.

    The C++ code stored `Location` as "pollutant" and `Pollutant` as "country", so
    the two categorical columns are the station index and the pollutant index.
    """
    raw = pd.DataFrame(
        {
            "lat": df["lat"],
            "lon": df["lon"],
            "value": df["value"],
            "location_idx": pd.factorize(df["location"].fillna(""))[0],
            "pollutant_idx": pd.factorize(df["pollutant"])[0],
        }
    )
    return pd.DataFrame(MinMaxScaler().fit_transform(raw), columns=raw.columns)


def correlations(X: pd.DataFrame) -> pd.DataFrame:
    return X[["lat", "lon", "value"]].corr()


def fit_kmeans(X: pd.DataFrame, k: int = 5, seed: int = 42) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)


def association_rules(X: pd.DataFrame, min_support: float = 0.1) -> pd.DataFrame:
    """Rules A => B between lat, lon and value, each discretised into three levels.

    Returns support, confidence and lift of every pair with support >= min_support.
    """
    bins = [-np.inf, 0.33, 0.66, np.inf]
    levels = ["low", "medium", "high"]
    cols = ["lat", "lon", "value"]
    items = pd.DataFrame(
        {c: c + "=" + pd.cut(X[c], bins, right=False, labels=levels).astype(str) for c in cols}
    )

    n = len(items)
    single = {c: items[c].value_counts(normalize=True) for c in cols}
    rows = []
    for a, b in combinations(cols, 2):
        for (item_a, item_b), count in items.groupby([a, b]).size().items():
            support = count / n
            rows.append(
                {
                    "antecedent": item_a,
                    "consequent": item_b,
                    "support": support,
                    "confidence": support / single[a][item_a],
                    "lift": support / (single[a][item_a] * single[b][item_b]),
                }
            )
    rules = pd.DataFrame(rows)
    return rules[rules["support"] >= min_support].sort_values("lift", ascending=False, ignore_index=True)

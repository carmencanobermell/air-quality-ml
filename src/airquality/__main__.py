"""python -m airquality [--csv PATH] [--out DIR] [--seed N]"""
import argparse
import json
from pathlib import Path

import pandas as pd

from .analysis import association_rules, correlations, cpp_style_features, fit_kmeans
from .classify import robustness_checks, run_experiments
from .data import DEFAULT_CSV, clean_for_modelling, load_data


def _models_table(results: dict, key: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {key: name, "model": model, **metrics}
            for name, res in results.items()
            for model, metrics in res["models"].items()
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV, type=Path)
    parser.add_argument("--out", default=Path("results"), type=Path)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    pd.options.display.float_format = "{:.4f}".format

    df = load_data(args.csv)
    clean = clean_for_modelling(df)
    print(f"[INFO] {len(df)} rows loaded, {len(clean)} after dropping implausible readings and non-pollutants")

    # Exploratory part of the C++ project, kept for the one-to-one validation
    X = cpp_style_features(df)
    corr = correlations(X)
    km = fit_kmeans(X, k=5, seed=args.seed)
    sizes = pd.Series(km.labels_).value_counts().sort_index()
    rules = association_rules(X)
    print("\n[C++ REPLICA] Correlations\n", corr)
    print(f"\n[C++ REPLICA] K-Means k=5, inertia={km.inertia_:.2f}\n", sizes.to_string())
    print("\n[C++ REPLICA] Association rules\n", rules)

    results = run_experiments(df, seed=args.seed)
    print("\n[CLASSIFICATION] Leakage experiments\n", _models_table(results, "experiment").to_string(index=False))

    robustness = robustness_checks(clean, seed=args.seed)
    for label, checks in robustness.items():
        print(f"\n[ROBUSTNESS] label = {label}\n", _models_table(checks, "evaluation").to_string(index=False))

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_rows": len(df),
        "n_rows_clean": len(clean),
        "cpp_replica": {
            "correlation": corr.to_dict(),
            "kmeans": {"inertia": km.inertia_, "cluster_sizes": sizes.tolist()},
            "rules": rules.to_dict(orient="records"),
        },
        "classification": results,
        "robustness": robustness,
    }
    (args.out / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[INFO] Results written to {args.out / 'metrics.json'}")


if __name__ == "__main__":
    main()

"""Train, save, load and use the final "high pollution" model.

The model is the clean pipeline of experiment 3 (one-hot encoding + gradient boosting,
no `value` among the features) fitted on all the cleaned rows. Read the README section
"How much of the clean score is real?" before trusting it for new countries: it does not
generalise to them.

    python -m airquality.model [--csv PATH] [--out models/model.joblib]
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline

from .classify import CATEGORICAL, NUMERIC, _preprocessor, _threshold, global_label
from .data import DEFAULT_CSV, clean_for_modelling, load_data

FEATURES = NUMERIC + CATEGORICAL
DEFAULT_MODEL = Path("models") / "model.joblib"


def train(clean: pd.DataFrame, seed: int = 42) -> dict:
    """Fit on all rows of a cleaned DataFrame. Returns the model and its metadata."""
    pipe = make_pipeline(
        _preprocessor(NUMERIC, CATEGORICAL), HistGradientBoostingClassifier(random_state=seed)
    ).fit(clean[FEATURES], global_label(clean))
    return {
        "pipeline": pipe,
        "features": FEATURES,
        "value_threshold": float(_threshold(clean["value"])),  # what "high" means, in raw units
        "n_train": len(clean),
        "sklearn_version": sklearn.__version__,
    }


def save(bundle: dict, path: Path | str = DEFAULT_MODEL) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load(path: Path | str = DEFAULT_MODEL) -> dict:
    return joblib.load(path)


def predict(bundle: dict, records: list[dict]) -> list[dict]:
    """records: dicts with country, pollutant, lat, lon."""
    X = pd.DataFrame(records)[bundle["features"]]
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    return [{"high_pollution": bool(p > 0.5), "probability": float(p)} for p in proba]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV, type=Path)
    parser.add_argument("--out", default=DEFAULT_MODEL, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    clean = clean_for_modelling(load_data(args.csv))
    bundle = train(clean, seed=args.seed)
    print(f"[INFO] Model trained on {bundle['n_train']} rows, saved to {save(bundle, args.out)}")


if __name__ == "__main__":
    main()

"""Build the reference outputs of the original C++ implementation.

The C++ parser splits on ';' without handling quotes, so it misreads the 31 rows
whose `Location` field is quoted. Both versions are run on the CSV without them.

  1. python validation/build_cpp_reference.py csv  OUTPUT.csv
        -> writes the CSV without quoted rows (input for the C++ program).
  2. Compile main_ref.cpp together with the original .h files and run:
        ./main_ref OUTPUT.csv labels.txt > stdout.txt
  3. python validation/build_cpp_reference.py json stdout.txt labels.txt
        -> writes validation/cpp_reference.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from airquality.analysis import cpp_style_features  # noqa: E402
from airquality.data import DEFAULT_CSV, load_data  # noqa: E402

FILTERED_CSV = HERE / "data_without_quoted_rows.csv"


def write_filtered_csv(out: Path) -> None:
    lines = DEFAULT_CSV.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    out.write_text("".join([lines[0]] + [l for l in lines[1:] if '"' not in l]), encoding="utf-8")


def inertia(X: np.ndarray, labels: np.ndarray) -> float:
    """SSE around each cluster mean (the C++ centroids are not exported)."""
    return float(sum(((X[labels == c] - X[labels == c].mean(axis=0)) ** 2).sum() for c in np.unique(labels)))


def build_json(stdout_path: Path, labels_path: Path) -> None:
    text = Path(stdout_path).read_text(encoding="utf-8")
    num = r"(-?[\d.]+(?:e-?\d+)?)"
    grab = lambda pattern: float(re.search(pattern, text).group(1))  # noqa: E731

    X = cpp_style_features(load_data(FILTERED_CSV)).to_numpy()
    runs = [np.array(line.split(","), dtype=int) for line in Path(labels_path).read_text().splitlines()]
    ref = {
        "n_rows": int(grab(r"Datos cargados: (\d+)")),
        "correlation": {
            "lat_value": grab(rf"Latitud vs Valor: {num}"),
            "lon_value": grab(rf"Longitud vs Valor: {num}"),
            "lat_lon": grab(rf"Latitud vs Longitud: {num}"),
        },
        "rules_support": {
            m.group(1): float(m.group(2)) for m in re.finditer(r"^(\w+ & \w+) => soporte: " + num, text, re.M)
        },
        "positives": int(grab(r"Positivos: (\d+)")),
        "n_train": int(grab(r"Train: (\d+)")),
        "n_test": int(grab(r"Test: (\d+)")),
        "weights": [float(w) for w in re.search(r"Pesos:(.*)", text).group(1).split()],
        "metrics": {
            "precision": grab(rf"Precisión: {num}"),
            "recall": grab(rf"Recall: {num}"),
            "f1": grab(rf"F1-Score: {num}"),
        },
        "kmeans_inertias": [inertia(X, labels) for labels in runs],
    }
    (HERE / "cpp_reference.json").write_text(json.dumps(ref, indent=2, ensure_ascii=False))
    print("Written", HERE / "cpp_reference.json")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "csv":
        write_filtered_csv(Path(sys.argv[2]))
    elif len(sys.argv) == 4 and sys.argv[1] == "json":
        build_json(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        sys.exit(__doc__)

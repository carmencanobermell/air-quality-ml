"""Loading and cleaning of world_air_quality.csv."""
from pathlib import Path

import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "world_air_quality.csv"

COLUMNS = {
    "Country Code": "country",
    "Location": "location",
    "Coordinates": "coordinates",
    "Pollutant": "pollutant",
    "Unit": "unit",
    "Value": "value",
}

# Upper bounds for a physically plausible ambient reading, per (pollutant, unit).
# They are set well above the highest values ever recorded outdoors, so they only
# remove readings that are clearly broken (e.g. CO = 3.2 million µg/m³ in Turkey).
# Pairs not listed here are not pollutants (TEMPERATURE, RELATIVEHUMIDITY, UM003)
# or have an inconsistent unit (PM2.5 in ppm), and are dropped as well.
PLAUSIBLE_MAX = {
    ("PM2.5", "µg/m³"): 2_000,
    ("PM10", "µg/m³"): 5_000,
    ("PM1", "µg/m³"): 2_000,
    ("BC", "µg/m³"): 500,
    ("CO", "µg/m³"): 60_000,  # ~50 ppm
    ("CO", "ppm"): 50,
    ("NO2", "µg/m³"): 4_000,  # ~2 ppm
    ("NO2", "ppm"): 2,
    ("NO", "µg/m³"): 2_500,  # ~2 ppm
    ("NO", "ppm"): 2,
    ("NOX", "µg/m³"): 4_000,
    ("NOX", "ppm"): 2,
    ("O3", "µg/m³"): 2_000,  # ~1 ppm
    ("O3", "ppm"): 1,
    ("SO2", "µg/m³"): 5_000,  # ~2 ppm
    ("SO2", "ppm"): 2,
}


def load_data(path: Path | str = DEFAULT_CSV) -> pd.DataFrame:
    """Return a DataFrame with country, location, pollutant, unit, value, lat, lon.

    Same filtering as the original C++ loader: rows without valid coordinates or
    with a negative value are dropped. Use `clean_for_modelling` on top of this
    for the classification experiments.
    """
    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        usecols=list(COLUMNS),
        # "NA" is the country code of Namibia, not a missing value; only empty cells are
        keep_default_na=False,
        na_values=[""],
    ).rename(columns=COLUMNS)

    latlon = (
        df.pop("coordinates")
        .str.split(",", n=1, expand=True)
        .reindex(columns=[0, 1])
        .apply(pd.to_numeric, errors="coerce")
    )
    df["lat"], df["lon"] = latlon[0], latlon[1]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["lat", "lon", "value"])
    return df[df["value"] >= 0].reset_index(drop=True)


def clean_for_modelling(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only pollutant readings with a known unit and a physically plausible value."""
    limit = pd.Series(
        [PLAUSIBLE_MAX.get(key) for key in zip(df["pollutant"], df["unit"])],
        index=df.index,
        dtype=float,
    )
    keep = limit.notna() & (df["value"] <= limit)
    return df[keep].reset_index(drop=True)

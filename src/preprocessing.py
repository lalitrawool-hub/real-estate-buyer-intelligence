"""Load and prepare the supplied Parcl buyer and transaction CSV files."""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def load_data(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two source files, with actionable errors for absent/empty data."""
    folder = Path(data_dir) if data_dir else ROOT
    paths = [folder / "clients.csv", folder / "properties.csv"]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Data file is empty: {path}")
    clients, properties = (pd.read_csv(p) for p in paths)
    if clients.empty or properties.empty:
        raise ValueError("The client or property dataset contains no rows.")
    if "client_id" not in clients or "client_ref" not in properties:
        raise ValueError("Expected client_id in clients.csv and client_ref in properties.csv.")
    return clients, properties


def build_buyer_features(clients: pd.DataFrame, properties: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions per buyer and join observed, non-PII client fields."""
    client_fields = ["client_id", "client_type", "country", "region", "acquisition_purpose",
                     "satisfaction_score", "loan_applied", "referral_channel"]
    missing = [c for c in client_fields if c not in clients.columns]
    if missing:
        raise ValueError(f"Missing expected client columns: {', '.join(missing)}")
    p = properties.copy()
    if "sale_price" in p:
        p["sale_price"] = pd.to_numeric(p["sale_price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    if "floor_area_sqft" in p:
        p["floor_area_sqft"] = pd.to_numeric(p["floor_area_sqft"], errors="coerce")
    if "transaction_date" in p:
        p["transaction_date"] = pd.to_datetime(p["transaction_date"], errors="coerce", dayfirst=True)
    aggregations = {}
    for col, fn in [("listing_id", "count"), ("sale_price", "mean"), ("floor_area_sqft", "mean"),
                    ("unit_category", "nunique"), ("listing_status", "nunique"), ("transaction_date", "max")]:
        if col in p.columns:
            aggregations[col] = fn
    if not aggregations:
        raise ValueError("No usable transaction columns found in properties.csv.")
    tx = p.groupby("client_ref", dropna=True).agg(aggregations).rename(columns={
        "listing_id": "transaction_count", "sale_price": "mean_sale_price",
        "floor_area_sqft": "mean_floor_area_sqft", "unit_category": "unit_category_count",
        "listing_status": "listing_status_count", "transaction_date": "last_transaction_date",
    })
    tx.index.name = "client_id"
    result = clients[client_fields].drop_duplicates("client_id").set_index("client_id").join(tx, how="left")
    if "last_transaction_date" in result:
        result["days_since_last_transaction"] = (result["last_transaction_date"].max() - result["last_transaction_date"]).dt.days
        result = result.drop(columns="last_transaction_date")
    result["transaction_count"] = result.get("transaction_count", pd.Series(index=result.index, dtype=float)).fillna(0)
    result = result.reset_index()
    return result


def make_preprocessor(frame: pd.DataFrame, exclude=None) -> ColumnTransformer:
    """Create a reusable fitted-ready transformer; exclude IDs and target labels."""
    exclude = set(exclude or []) | {"client_id"}
    features = frame.drop(columns=[c for c in exclude if c in frame], errors="ignore")
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical))
    if not transformers:
        raise ValueError("No usable features remain after excluding identifiers.")
    return ColumnTransformer(transformers, remainder="drop")

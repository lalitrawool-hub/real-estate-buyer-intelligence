"""Readable summaries of observed buyer segments."""
import pandas as pd


def summarize_segments(frame: pd.DataFrame, label_col="segment") -> pd.DataFrame:
    if label_col not in frame:
        raise ValueError(f"'{label_col}' column is required for segment profiling.")
    rows = []
    numeric = frame.select_dtypes(include="number").columns.tolist()
    for label, group in frame.groupby(label_col, dropna=False):
        row = {"segment": str(label), "buyers": len(group)}
        for col in numeric:
            if col != label_col:
                row[f"{col}_avg"] = group[col].mean()
        for col in ("client_type", "country", "region", "acquisition_purpose", "loan_applied", "referral_channel"):
            if col in group:
                mode = group[col].mode(dropna=True)
                row[f"top_{col}"] = mode.iloc[0] if not mode.empty else "Unknown"
        row["share_of_buyers_pct"] = 100 * len(group) / max(len(frame), 1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("buyers", ascending=False).reset_index(drop=True)

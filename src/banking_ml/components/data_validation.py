import pandas as pd

class DataValidation:
    @staticmethod
    def validate(df: pd.DataFrame, target: str, required_columns=None):
        if df.empty:
            raise ValueError("Dataset is empty.")
        if df.columns.duplicated().any():
            raise ValueError("Duplicate column names detected.")
        required = set(required_columns or []) | {target}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if df[target].isna().any():
            raise ValueError(f"Target column '{target}' contains null values.")
        if df[target].nunique() < 2:
            raise ValueError("Classification target must contain at least two classes.")
        return {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "target_distribution": df[target].value_counts(dropna=False).to_dict(),
        }

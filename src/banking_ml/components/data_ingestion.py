from pathlib import Path
import pandas as pd

class DataIngestion:
    def __init__(self, source_path: str):
        self.source_path = Path(source_path)

    def load(self) -> pd.DataFrame:
        if not self.source_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.source_path}")
        suffix = self.source_path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(self.source_path)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(self.source_path)
        raise ValueError(f"Unsupported file type: {suffix}. Add a connector for this source.")

import argparse
from pathlib import Path
import joblib
import pandas as pd
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV file containing new observations")
    parser.add_argument("--output", default="data/predictions/scored_customers.csv")
    args = parser.parse_args()
    cfg = yaml.safe_load(open("config/config.yaml", encoding="utf-8"))
    model = joblib.load(cfg["training"]["model_path"])
    df = pd.read_csv(args.input)
    excluded = set(cfg["data"].get("id_columns", [])) | set(cfg["data"].get("drop_columns", []))
    if cfg["data"].get("date_column"):
        excluded.add(cfg["data"]["date_column"])
    X = df.drop(columns=[c for c in excluded if c in df.columns], errors="ignore")
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    positive = cfg["data"]["positive_label"]
    pos_idx = classes.index(positive)
    df["probability_positive"] = probabilities[:, pos_idx]
    df["predicted_class"] = model.predict(X)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")

if __name__ == "__main__":
    main()

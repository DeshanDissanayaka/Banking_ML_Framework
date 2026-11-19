from pathlib import Path
import joblib
import pandas as pd
from banking_ml.components.data_ingestion import DataIngestion
from banking_ml.components.data_validation import DataValidation
from banking_ml.components.feature_engineering import FeatureEngineering
from banking_ml.components.data_splitting import DataSplitting
from banking_ml.components.model_trainer import ModelTrainer
from banking_ml.components.model_evaluation import ModelEvaluation
from banking_ml.utils.logger import get_logger

class TrainingPipeline:
    def __init__(self, config, params):
        self.config, self.params = config, params
        self.logger = get_logger(log_path=config["logging"]["path"])

    def run(self):
        cfg = self.config
        df = DataIngestion(cfg["data"]["path"]).load()
        report = DataValidation.validate(df, cfg["data"]["target"])
        self.logger.info("Data validation: %s", report)

        df = FeatureEngineering().transform(df)
        target = cfg["data"]["target"]
        excluded = set(cfg["data"].get("id_columns", []))
        excluded.update(cfg["data"].get("drop_columns", []))
        if cfg["data"].get("date_column"):
            excluded.add(cfg["data"]["date_column"])
        excluded.add(target)
        X = df.drop(columns=[c for c in excluded if c in df.columns])
        y = df[target]

        if X.empty:
            raise ValueError("No feature columns remain after exclusions.")
        parts = DataSplitting.split(
            X, y, cfg["split"]["test_size"], cfg["split"]["validation_size"],
            cfg["project"]["random_state"]
        )
        X_train, X_val, X_test, y_train, y_val, y_test = parts
        model_name = cfg["training"]["model"]
        model_params = self.params["models"][model_name]
        trainer = ModelTrainer()
        pipeline = trainer.build_pipeline(X_train, model_name, model_params,
            self.params["preprocessing"], cfg["project"]["random_state"])
        pipeline = trainer.fit(pipeline, X_train, y_train)

        evaluator = ModelEvaluation()
        val_metrics, _, _ = evaluator.evaluate(pipeline, X_val, y_val,
            cfg["training"]["threshold"], cfg["data"]["positive_label"])
        test_metrics, probabilities, predictions = evaluator.evaluate(pipeline, X_test, y_test,
            cfg["training"]["threshold"], cfg["data"]["positive_label"])
        self.logger.info("Validation F1: %.4f | Test F1: %.4f",
            val_metrics["f1"], test_metrics["f1"])

        model_path = Path(cfg["training"]["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)
        evaluator.save({"validation": val_metrics, "test": test_metrics},
                       cfg["training"]["metrics_path"])
        pred_path = Path(cfg["training"]["predictions_path"])
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"actual": y_test.to_numpy(), "prediction": predictions,
                      "probability_positive": probabilities}).to_csv(pred_path, index=False)
        self._save_model_card(model_path, model_name, cfg, test_metrics)
        return test_metrics

    @staticmethod
    def _save_model_card(model_path, model_name, cfg, metrics):
        card = {
            "project": cfg["project"]["name"], "task": cfg["project"]["task"],
            "model": model_name, "target": cfg["data"]["target"],
            "threshold": cfg["training"]["threshold"],
            "test_metrics": metrics,
            "limitations": [
                "Baseline split is random and may be unsuitable for temporal or repeated-entity data.",
                "Metrics are dataset-specific and do not establish production readiness.",
                "Probability calibration, fairness, stability and independent validation are not included."
            ]
        }
        path = Path(cfg["training"]["artifact_dir"]) / "model_card.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        path.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")

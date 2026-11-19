# Learning guide

Study in this order; do not start with Docker or Kubernetes.

1. Python packaging: modules, packages, classes, imports, setup.py.
2. YAML configuration: separate settings from code.
3. Data ingestion: CSV/Parquet first, then SQL/API adapters.
4. Data quality: missingness, duplicates, schema, target leakage.
5. EDA: distributions, relationships, outliers, imbalance.
6. Feature engineering: domain logic and point-in-time correctness.
7. Splitting: random vs temporal vs customer-grouped validation.
8. Preprocessing: imputation, encoding, scaling, sklearn Pipeline.
9. Baselines: logistic regression before complex models.
10. Models: random forest, extra trees, gradient boosting / XGBoost / LightGBM.
11. Evaluation: PR-AUC, ROC-AUC, recall, precision, F1, calibration, costs.
12. Explainability: coefficients, permutation importance, SHAP.
13. Tuning: cross-validation and Optuna; tune only within training data.
14. Experiment tracking: MLflow parameters, metrics, artifacts.
15. Reproducibility: DVC, Git commit, data version and environment.
16. Inference: batch scoring and FastAPI.
17. Testing: unit, integration, data contract and prediction tests.
18. Deployment: Docker, CI/CD, secrets and observability.
19. Monitoring: drift, performance, calibration, fairness and retraining triggers.

For banking, add model governance: business rationale, intended use, limitations,
independent validation, fairness assessment, explainability, approval and audit trail.

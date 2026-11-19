# Architecture and design

## Layer responsibilities
1. Data sources: adapters for CSV, Parquet, SQL and APIs.
2. Components: small, testable data/ML operations.
3. Pipelines: orchestrate components in a repeatable order.
4. Models: task-specific estimators and factories.
5. API: serves a persisted, versioned inference pipeline.
6. MLOps: tracks data, code, parameters, artifacts and deployment.

## Training flow
Raw source -> validation -> point-in-time-safe feature engineering -> split ->
preprocessing/model pipeline -> validation/test evaluation -> artifact + metrics + model card.

## Key design decision
Preprocessing is inside the scikit-learn Pipeline. Therefore imputers, encoders and
scalers are fitted on training data and reused unchanged during inference. This prevents
a common train/inference mismatch and reduces preprocessing leakage.

## Production extensions
- SQL connector with secrets manager and read-only credentials
- Great Expectations or Pandera data contracts
- Time-based and group-aware split strategies
- Optuna for tuning
- MLflow for experiment tracking and model registry
- DVC for data/pipeline versioning
- SHAP for explainability
- Evidently or custom PSI/KS monitoring
- model approval workflow, champion/challenger and rollback
- authentication, authorization, rate limits and audit logs for API

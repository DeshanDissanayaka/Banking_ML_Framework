# Banking ML & Data Science Framework

Reusable, configuration-driven starter framework for tabular banking ML projects.

## Supported project families
- Classification (default, delinquency, churn, fraud baseline)
- Regression (loss, exposure, customer value)
- Clustering (segmentation)
- Forecasting (time-aware extensions)

The included executable baseline is **binary classification** using scikit-learn. Other problem families have extension points and should not be treated as implemented merely because folders exist.

## Quick start
1. Create a virtual environment (Python 3.10–3.12 recommended).
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and edit paths if needed.
4. Put a CSV dataset in `data/raw/`.
5. Edit `config/config.yaml` and `config/params.yaml`.
6. Set the target column and positive class in `config/config.yaml`.
7. Run `python scripts/train.py`
8. Run `python scripts/predict.py --input data/raw/new_customers.csv`

The training script creates a stratified train/validation/test split, fits preprocessing only on training data through an sklearn Pipeline, evaluates the held-out test set, saves the fitted pipeline, metrics and a model card.

## First project: loan default prediction
Example target: `default_flag` (0 = non-default, 1 = default).
Use a point-in-time feature table: every feature must be available as of the prediction date. Exclude post-outcome fields, collections actions after default, write-off amounts recorded later, and identifiers that leak the label.

## Change for a new project
Usually change:
- `config/config.yaml`: source, target, task, ID/date columns, positive label
- `config/params.yaml`: split and model parameters
- `config/schema.yaml`: expected columns and basic constraints
- `src/banking_ml/components/feature_engineering.py`: project-specific, point-in-time-safe features
- `src/banking_ml/models/`: algorithm candidates or task-specific model logic

Keep reusable ingestion, validation, splitting, evaluation, logging, API and deployment modules stable.

## Important safeguards
- Never use customer names, account numbers, NICs or other direct identifiers as predictive features.
- Split by time for temporal deployment problems; random stratification is only a baseline.
- For repeated customer observations, consider customer-grouped splitting to prevent entity leakage.
- Fit imputers, encoders and scalers on training data only. The included sklearn Pipeline does this.
- For rare fraud/default events, report PR-AUC, recall, precision, F1 and confusion matrix; accuracy alone is inadequate.
- Choose thresholds based on business costs, capacity and risk appetite, not automatically 0.5.
- Calibrate probabilities before treating them as PD estimates; validate calibration and population stability.
- Do not deploy a model for credit decisions without governance, fairness, explainability, independent validation and human oversight.
- Synthetic/demo data and metrics are not evidence of real-world performance.

## Project structure
See `docs/ARCHITECTURE.md` and `docs/LEARNING_GUIDE.md`.

## Reproducibility
Set `random_state` in configuration. Record dataset version/hash, code commit, config, package versions and model artifact together. DVC and MLflow are optional integrations to add after the baseline is understood.

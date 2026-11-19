from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from banking_ml.models.classification.model_factory import build_classifier

class ModelTrainer:
    def build_pipeline(self, X, model_name, model_params, preprocessing, random_state=42):
        numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical = [c for c in X.columns if c not in numeric]
        numeric_steps = [("imputer", SimpleImputer(strategy=preprocessing.get("numeric_strategy", "median")))]
        if preprocessing.get("scale_numeric", False):
            numeric_steps.append(("scaler", StandardScaler()))
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=preprocessing.get("categorical_strategy", "most_frequent"))),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformer = ColumnTransformer([
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", categorical_pipe, categorical),
        ], remainder="drop")
        model = build_classifier(model_name, model_params, random_state)
        return Pipeline([("preprocessing", transformer), ("model", model)])

    def fit(self, pipeline, X_train, y_train):
        pipeline.fit(X_train, y_train)
        return pipeline

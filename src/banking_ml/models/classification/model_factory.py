from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

def build_classifier(model_name: str, params: dict, random_state: int = 42):
    if model_name == "logistic_regression":
        return LogisticRegression(random_state=random_state, **params)
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=random_state, **params)
    if model_name == "extra_trees":
        return ExtraTreesClassifier(random_state=random_state, **params)
    raise ValueError(f"Unknown classifier: {model_name}")

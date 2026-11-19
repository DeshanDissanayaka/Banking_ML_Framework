import pandas as pd

class FeatureEngineering:
    """Project-specific feature logic belongs here.

    Keep features point-in-time safe: a feature must have existed at the
    prediction timestamp. Do not use post-default/post-fraud information.
    """
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        # Example extension:
        # result["debt_to_income"] = result["monthly_debt"] / result["monthly_income"].clip(lower=1)
        return result

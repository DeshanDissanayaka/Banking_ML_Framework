import pandas as pd
import pytest
from banking_ml.components.data_validation import DataValidation

def test_validation_accepts_binary_target():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    result = DataValidation.validate(df, "target")
    assert result["rows"] == 4

def test_validation_rejects_missing_target():
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError):
        DataValidation.validate(df, "target")

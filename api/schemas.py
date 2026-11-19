from pydantic import BaseModel
from typing import Any

class PredictionRequest(BaseModel):
    records: list[dict[str, Any]]

class PredictionResponse(BaseModel):
    predictions: list[dict[str, Any]]

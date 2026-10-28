"""Pydantic-схемы API (валидация типов, допустимых значений и диапазонов)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "customerID": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
            "Dependents": "No", "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
            "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85,
            "TotalCharges": 29.85,
        }},
    )

    customerID: Optional[str] = Field(None, max_length=64, description="Необязательный ID — вернётся в ответе")
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(..., ge=0, le=120, description="Месяцев с компанией")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(..., ge=0, le=500)
    TotalCharges: Optional[float] = Field(None, ge=0, le=100_000,
                                          description="Можно не передавать (новый клиент) — будет MonthlyCharges*tenure")


class Prediction(BaseModel):
    customerID: Optional[str] = None
    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: Literal[0, 1]
    risk_segment: Literal["low", "medium", "high"]

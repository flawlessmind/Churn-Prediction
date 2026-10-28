import json

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.api import app
from src.data import prepare_xy
from src.features import ChurnFeatureEngineer

PAYLOAD = {
    "customerID": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85,
}


@pytest.fixture(scope="module")
def client(raw_df, tmp_path_factory):
    X, y = prepare_xy(raw_df)
    pipe = Pipeline([("fe", ChurnFeatureEngineer()), ("scaler", StandardScaler()),
                     ("model", LogisticRegression(max_iter=1000))]).fit(X, y)
    d = tmp_path_factory.mktemp("model")
    joblib.dump(pipe, d / "churn_model.pkl")
    (d / "model_metadata.json").write_text(json.dumps({
        "model_name": "test", "model_version": "0.0.1", "trained_at": "2026-01-01T00:00:00+00:00",
        "thresholds": {"f1_opt": 0.35, "top20_budget": 0.6}, "holdout_metrics": {"pr_auc": 0.5},
        "n_train": 10, "n_holdout": 5}))
    import os
    os.environ["MODEL_PATH"] = str(d / "churn_model.pkl")
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict(client):
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200, r.text
    b = r.json()
    assert 0 <= b["churn_probability"] <= 1 and b["churn_prediction"] in (0, 1)
    assert b["risk_segment"] in ("low", "medium", "high") and b["customerID"] == "7590-VHVEG"


def test_predict_without_total_charges(client):
    p = {k: v for k, v in PAYLOAD.items() if k != "TotalCharges"}
    assert client.post("/predict", json=p).status_code == 200


def test_predict_batch(client):
    r = client.post("/predict_batch", json=[PAYLOAD, {**PAYLOAD, "Contract": "Two year", "tenure": 60}])
    assert r.status_code == 200 and len(r.json()) == 2


def test_validation_errors_422(client):
    assert client.post("/predict", json={**PAYLOAD, "tenure": -1}).status_code == 422
    assert client.post("/predict", json={**PAYLOAD, "Contract": "Weekly"}).status_code == 422
    assert client.post("/predict", json={k: v for k, v in PAYLOAD.items() if k != "gender"}).status_code == 422
    assert client.post("/predict_batch", json=[]).status_code == 422


def test_model_info(client):
    r = client.get("/model_info")
    assert r.status_code == 200 and r.json()["model_version"] == "0.0.1"

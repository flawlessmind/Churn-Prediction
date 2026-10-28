"""Синтетический Telco-подобный датасет. ТОЛЬКО для тестов/CI/smoke-проверки пайплайна без Kaggle.
Метрики на нём ничего не говорят о качестве модели на реальных данных."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_telco(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pick = lambda opts, p=None: rng.choice(opts, size=n, p=p)  # noqa: E731

    internet = pick(["DSL", "Fiber optic", "No"], [0.35, 0.45, 0.20])
    phone = pick(["Yes", "No"], [0.9, 0.1])
    multiple = np.where(phone == "No", "No phone service", pick(["No", "Yes"]))
    addons = {
        c: np.where(internet == "No", "No internet service", pick(["No", "Yes"]))
        for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    }
    contract = pick(["Month-to-month", "One year", "Two year"], [0.55, 0.21, 0.24])
    payment = pick(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    tenure = np.clip(rng.exponential(30, n), 0, 72).astype(int)
    tenure[rng.random(n) < 0.01] = 0

    n_addons = sum((v == "Yes").astype(int) for v in addons.values())
    monthly = np.round(
        20 + 12 * (internet == "DSL") + 35 * (internet == "Fiber optic") + 5 * n_addons
        + 8 * (multiple == "Yes") + rng.normal(0, 3, n), 2,
    ).clip(18.25, 118.75)
    total = np.round(monthly * tenure * rng.uniform(0.95, 1.05, n), 2)

    logit = (
        -0.8 + 1.3 * (contract == "Month-to-month") - 0.9 * (contract == "Two year")
        - 0.035 * tenure + 0.9 * (internet == "Fiber optic") + 0.5 * (payment == "Electronic check")
        - 0.6 * (addons["TechSupport"] == "Yes") - 0.4 * (addons["OnlineSecurity"] == "Yes")
        + 0.01 * (monthly - 65)
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    return pd.DataFrame({
        "customerID": [f"{i:04d}-SYNTH" for i in range(n)],
        "gender": pick(["Female", "Male"]),
        "SeniorCitizen": pick([0, 1], [0.84, 0.16]),
        "Partner": pick(["Yes", "No"]),
        "Dependents": pick(["Yes", "No"], [0.3, 0.7]),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        **addons,
        "Contract": contract,
        "PaperlessBilling": pick(["Yes", "No"], [0.6, 0.4]),
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        # как в оригинале: TotalCharges — строка, у tenure=0 — пробел
        "TotalCharges": [" " if t == 0 else str(v) for t, v in zip(tenure, total)],
        "Churn": np.where(churn, "Yes", "No"),
    })

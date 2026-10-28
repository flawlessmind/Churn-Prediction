.PHONY: install data synthetic eda train explain serve test docker-train docker-up mlflow-ui
install:        ; pip install -r requirements.txt
data:           ; python scripts/download_data.py
synthetic:      ; python scripts/make_synthetic_data.py      # только для smoke-теста без Kaggle
eda:            ; python -m src.eda
train:          ; python -m src.train --n-trials 30
explain:        ; python -m src.explain
serve:          ; uvicorn src.api:app --reload --port 8000
test:           ; pytest -q
docker-train:   ; docker compose --profile train run --rm trainer
docker-up:      ; docker compose up --build
mlflow-ui:      ; mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db

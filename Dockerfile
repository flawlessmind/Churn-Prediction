# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11

# ---------- build: venv с зависимостями inference ----------
FROM python:${PYTHON_VERSION}-slim AS builder-serve
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# ---------- build: + зависимости обучения (optuna, mlflow, shap, ...) ----------
FROM builder-serve AS builder-train
COPY requirements-train.txt .
RUN pip install -r requirements-train.txt

# ---------- trainer: образ для обучения (docker compose --profile train run trainer) ----------
FROM python:${PYTHON_VERSION}-slim AS trainer
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=builder-train /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY src ./src
COPY scripts ./scripts
CMD ["sh", "-c", "python -m src.train && python -m src.explain"]

# ---------- runtime: только то, что нужно для сервиса (последний стейдж = образ по умолчанию) ----------
FROM python:${PYTHON_VERSION}-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
COPY --from=builder-serve /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/churn_model.pkl
WORKDIR /app
COPY src ./src
COPY models ./models
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

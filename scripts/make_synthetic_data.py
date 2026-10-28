"""Генерирует СИНТЕТИЧЕСКИЙ Telco-подобный CSV — только чтобы проверить пайплайн без Kaggle."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as cfg  # noqa: E402
from src.synthetic import make_synthetic_telco  # noqa: E402

if __name__ == "__main__":
    cfg.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    make_synthetic_telco(n=7043, seed=cfg.RANDOM_STATE).to_csv(cfg.RAW_CSV, index=False)
    print(f"Синтетические данные записаны в {cfg.RAW_CSV}\nВНИМАНИЕ: это не реальный датасет, метрики ничего не значат.")

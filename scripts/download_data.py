"""Скачивает Telco Customer Churn с Kaggle (blastchar/telco-customer-churn) в data/raw/.
Нужен токен: https://www.kaggle.com/settings -> Create New Token -> ~/.kaggle/kaggle.json"""
import subprocess
import sys
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "data" / "raw"

if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", "blastchar/telco-customer-churn", "-p", str(DEST), "--unzip"]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("Не найден CLI `kaggle`. Установите: pip install kaggle (входит в requirements-train.txt).")
    except subprocess.CalledProcessError:
        sys.exit("Не удалось скачать датасет. Проверьте ~/.kaggle/kaggle.json (или KAGGLE_USERNAME/KAGGLE_KEY) "
                 "либо скачайте CSV вручную со страницы датасета в data/raw/.")
    print("Готово:", *DEST.glob("*.csv"), sep="\n  ")

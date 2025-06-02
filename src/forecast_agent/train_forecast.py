import os
import pandas as pd

from forecast_agent import ForecastAgent

if __name__ == "__main__":
    # Шлях до підготовлених CSV
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed"))
    train_csv = os.path.join(base_dir, "train_prepared.csv")
    val_csv = os.path.join(base_dir, "validation.csv")

    # Шляхи для збереження моделі та метрик
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "forecast_model.pkl")
    metrics_path = os.path.join(models_dir, "metrics.json")

    print("[train_forecast] Починаємо тренування ForecastAgent...")
    fa = ForecastAgent()
    fa.train(train_csv=train_csv,
             val_csv=val_csv,
             model_out_path=model_path,
             metrics_out_path=metrics_path)
    print("[train_forecast] Тренування завершено.")

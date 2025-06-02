import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestRegressor

class ForecastAgent:
    """
    Проста модель прогнозування продажів на основі RandomForestRegressor.
    Зчитує попередньо підготовлені CSV (train_prepared.csv, validation.csv) з data/processed,
    навчає модель, зберігає її у models/forecast_model.pkl; прогноз формує через model.predict(X).
    """

    def __init__(self, model_path: str = None):
        """
        Якщо передано model_path, завантажує модель із цього шляху; 
        інакше встановлює self.model = None.
        """
        self.model = None
        if model_path:
            self.load_model(model_path)

    def train(self,
              train_csv: str,
              val_csv: str,
              model_out_path: str,
              metrics_out_path: str) -> None:
        """
        1) Зчитує train_prepared.csv і validation.csv
        2) Відокремлює X_train, y_train, X_val, y_val
        3) Навчає RandomForestRegressor
        4) Обчислює MAE, RMSE на валідації
        5) Зберігає модель у model_out_path (joblib), а метрики у metrics_out_path (JSON)
        """
        import json
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        # Зчитуємо підготовлені дані
        df_train = pd.read_csv(train_csv)
        df_val   = pd.read_csv(val_csv)

        # Цільова: "Sales"
        y_train = df_train["Sales"].values
        X_train = df_train.drop(columns=["Sales", "Date"]).values

        y_val = df_val["Sales"].values
        X_val = df_val.drop(columns=["Sales", "Date"]).values

        # Навчання RandomForest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        # Прогноз на валідації
        preds_val = rf.predict(X_val)
        mae = mean_absolute_error(y_val, preds_val)
        mse = mean_squared_error(y_val, preds_val)
        rmse = mse ** 0.5

        # Збереження моделі
        os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
        joblib.dump(rf, model_out_path)

        # Збереження метрик
        metrics = {"MAE": float(mae), "RMSE": float(rmse)}
        with open(metrics_out_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[ForecastAgent.train] Модель збережена у {model_out_path}")
        print(f"[ForecastAgent.train] Метрики валідації: MAE = {mae:.2f}, RMSE = {rmse:.2f}")
        print(f"[ForecastAgent.train] Метрики збережено у {metrics_out_path}")

        # Заносимо модель у self.model
        self.model = rf

    def load_model(self, model_path: str) -> None:
        """
        Завантажує модель із диску (joblib .pkl або .joblib).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель за шляхом {model_path} не знайдена.")
        self.model = joblib.load(model_path)
        print(f"[ForecastAgent.load_model] Модель завантажена з {model_path}")

    def predict(self,
                store_id: int,
                start_date: pd.Timestamp,
                horizon_days: int,
                store_csv: str = None) -> np.ndarray:
        """
        Побудова фічей для кожного дня з start_date на горизонті horizon_days
        і повернення масиву прогнозованих Sales.

        Щоб ПОВНІСТЮ відповідати тренувальному датасету (22 вхідні колонки), 
        сюди ми додаємо дами для:
          • StoreType_b, StoreType_c, StoreType_d   (базова в тренуванні – StoreType_a)
          • Assortment_b, Assortment_c               (базова – Assortment_a)
          • DayOfWeek_2, …, DayOfWeek_7              (базова – DayOfWeek_1)
        
        У підсумку: 11 базових числових + 3 StoreType‐дами + 2 Assortment‐дами + 6 DayOfWeek‐дам = 22 колонки.
        """
        import datetime

        if self.model is None:
            raise RuntimeError("Модель не завантажена. Викличте load_model() або train().")

        # Для того, щоб дістати атрибути магазину (конкуренція, тип тощо),
        # потрібно передати параметр store_csv (raw store.csv)
        if store_csv is None:
            raise ValueError("Для побудови фічей потрібен параметр store_csv (raw store.csv).")

        df_store = pd.read_csv(store_csv)
        # Якщо магазинів у store_csv менше, ніж у повному датасеті, 
        # для тих, яких нема, store_info буде порожнім—але припустимо, що ви обираєте реальний store_id із датасету.
        store_info = df_store[df_store["Store"] == store_id]
        if store_info.shape[0] == 0:
            raise KeyError(f"Магазин з ID={store_id} не знайдено в {store_csv}")
        store_info = store_info.iloc[0]

        # Збираємо записи для кожного дня горизонту
        records = []
        for offset in range(horizon_days):
            dt = start_date + datetime.timedelta(days=offset)

            rec = {
                # ‣ Базові числові колонки:
                "Store": store_id,
                "Year": dt.year,
                "Month": dt.month,
                "Day": dt.day,
                "Customers": 0,         # невідомо заздалегідь → 0
                "Promo": 0,             # прогнози промо за майбутнє невідомі → 0
                "SchoolHoliday": 0,     # задамо 0
                "IsHoliday": 0,         # задамо 0
                "CompetitionDistance": store_info["CompetitionDistance"],
                "CompetitionOpenSinceMonth": int(store_info["CompetitionOpenSinceMonth"]) 
                                            if not np.isnan(store_info["CompetitionOpenSinceMonth"]) else 0,
                "CompetitionOpenSinceYear":  int(store_info["CompetitionOpenSinceYear"])  
                                            if not np.isnan(store_info["CompetitionOpenSinceYear"])  else 0,
            }

            # --- Dummy для StoreType: generуємо колонки для b, c, d; 
            # базова «a» означає, що всі три будуть 0
            for st in ["b", "c", "d"]:
                rec[f"StoreType_{st}"] = 1 if store_info["StoreType"] == st else 0

            # --- Dummy для Assortment: generуємо колонки для b, c; базова «a» → обидва 0
            for a in ["b", "c"]:
                rec[f"Assortment_{a}"] = 1 if store_info["Assortment"] == a else 0

            # --- Dummy для DayOfWeek: generуємо DayOfWeek_2 … DayOfWeek_7; 
            # DayOfWeek_1 – базова (якщо dt.weekday()+1 == 1, то всі ці дамі = 0)
            dow = dt.weekday() + 1  # DayOfWeek у даних – від 1 до 7
            for d in range(2, 8):
                rec[f"DayOfWeek_{d}"] = 1 if dow == d else 0

            records.append(rec)

        df_pred = pd.DataFrame(records)

        # Фіксуємо порядок ознак точно так само, як під час тренування:
        feature_cols = [
            # 1) ‣ Базові числові:
            "Store",
            "Year", "Month", "Day",
            "Customers", "Promo", "SchoolHoliday", "IsHoliday",
            "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
            # 2) ‣ Dummy-сті для StoreType:
            "StoreType_b", "StoreType_c", "StoreType_d",
            # 3) ‣ Dummy-сті для Assortment:
            "Assortment_b", "Assortment_c",
            # 4) ‣ Dummy-сті для DayOfWeek:
            "DayOfWeek_2", "DayOfWeek_3", "DayOfWeek_4",
            "DayOfWeek_5", "DayOfWeek_6", "DayOfWeek_7"
        ]

        # Перевіряємо, чи всі колонки присутні:
        missing = [col for col in feature_cols if col not in df_pred.columns]
        if missing:
            raise KeyError(f"У DataFrame для прогнозу не знайдено необхідні стовпці: {missing}")

        X_pred = df_pred[feature_cols].values
        # DEBUG-друк (переконайтеся, що це (n_samples, 22))
        print("DEBUG: X_pred shape =", X_pred.shape)

        preds = self.model.predict(X_pred)
        return preds

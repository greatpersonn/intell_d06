import pandas as pd
import numpy as np
import os

from load_data import load_train, load_store

def preprocess_and_save(raw_dir: str, processed_dir: str):
    """
    1) Завантажує train.csv та store.csv
    2) Об’єднує їх
    3) Створює додаткові фічі (year, month, day, day_of_week, is_holiday тощо)
    4) Закодовує категоріальні змінні (StoreType, Assortment) через one-hot
    5) Видаляє дні, коли магазин був зачинений (Open == 0)
    6) Розбиває на train/validation за датою (останні 6 тижнів як валідація)
    7) Зберігає готові CSV у data/processed
    """
    # Створюємо папку processed, якщо нема
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # 1. Завантаження
    df_train = load_train(raw_dir)
    df_store = load_store(raw_dir)

    # 2. Об’єднати за "Store"
    df = pd.merge(df_train, df_store, on="Store", how="left")

    # 3. Фічі з дати
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["DayOfWeek"].astype(str)  # зробимо string, щоб one-hot пішов
    df["IsHoliday"] = np.where(df["StateHoliday"] != "0", 1, 0)

    # 4. Видалити дні, коли Open == 0
    df = df[df["Open"] == 1].copy()

    # 5. One-hot для StoreType та Assortment
    df["StoreType"] = df["StoreType"].astype(str)
    df["Assortment"] = df["Assortment"].astype(str)
    df = pd.get_dummies(df, columns=["StoreType", "Assortment", "DayOfWeek"], drop_first=True)

    # 6. Вибрані фічі
    features = [
        "Store",
        "Year", "Month", "Day",
        "Customers", "Promo", "SchoolHoliday", "IsHoliday",
        "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    ]
    # Додаємо всі нові даммі: StoreType_*, Assortment_*, DayOfWeek_*
    dummy_cols = [col for col in df.columns if col.startswith("StoreType_") or col.startswith("Assortment_") or col.startswith("DayOfWeek_")]
    features += dummy_cols

    # В результаті нам потрібен датафрейм з цими фічами та цільовою змінною Sales
    df_final = df[features + ["Sales", "Date"]].copy()

    # Видаляємо рядки з пропусками (якщо є)
    df_final = df_final.dropna().reset_index(drop=True)

    # 7. Розбиваємо на train / validation
    # Візьмемо останні 6 тижнів (42 дні) як validation
    cutoff_date = df_final["Date"].max() - pd.Timedelta(days=42)
    df_train_prepared = df_final[df_final["Date"] <= cutoff_date].copy()
    df_val_prepared = df_final[df_final["Date"] > cutoff_date].copy()

    # Зберігаємо
    train_path = os.path.join(processed_dir, "train_prepared.csv")
    val_path = os.path.join(processed_dir, "validation.csv")
    df_train_prepared.to_csv(train_path, index=False)
    df_val_prepared.to_csv(val_path, index=False)

    print(f"[preprocess] Підготовлені дані збережено у:\n  {train_path}\n  {val_path}")


if __name__ == "__main__":
    raw_directory = os.path.join(os.path.dirname(__file__), "../../data/raw")
    processed_directory = os.path.join(os.path.dirname(__file__), "../../data/processed")
    # Уточнюємо шлях
    raw_directory = os.path.abspath(raw_directory)
    processed_directory = os.path.abspath(processed_directory)

    preprocess_and_save(raw_directory, processed_directory)

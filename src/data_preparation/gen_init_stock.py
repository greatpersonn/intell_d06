import pandas as pd
import numpy as np
import os

def generate_initial_stock(raw_dir: str, output_path: str, low: int = 300, high: int = 500):
    """
    Згенерувати початкові запаси (300–500 одиниць) для кожного магазину,
    взявши список унікальних Store з train.csv або store.csv.
    Зберегти у CSV: columns = ["Store", "InitialStock"]
    """
    # Читаємо унікальні ідентифікатори магазинів
    stores = pd.read_csv(os.path.join(raw_dir, "store.csv"))["Store"].unique()
    # Генеруємо випадковий сток
    np.random.seed(42)
    initial_stock = np.random.randint(low, high + 1, size=len(stores))

    df_stock = pd.DataFrame({"Store": stores, "InitialStock": initial_stock})
    df_stock.to_csv(output_path, index=False)
    print(f"[generate_initial_stock] Початкові запаси збережено у {output_path}")


if __name__ == "__main__":
    raw_directory = os.path.join(os.path.dirname(__file__), "../../data/raw")
    raw_directory = os.path.abspath(raw_directory)
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed/initial_stock.csv"))
    generate_initial_stock(raw_directory, out_path)

import pandas as pd
from datetime import date, timedelta
import os

from forecast_agent.forecast_agent import ForecastAgent
from inventory_agent.inventory_agent import InventoryAgent
from supplier_agent.supplier_agent import SupplierAgent
from utils.calculate_metrics import calculate_total_cost, calculate_fill_rate

def main():
    """
    1) Завантажити попередньо збережену модель ForecastAgent
    2) Ініціалізувати InventoryAgent (зі згенерованими початковими запасами)
    3) Ініціалізувати SupplierAgent
    4) Запустити цикл симуляції з 2025-01-01 по 2025-03-31 із кроком у 7 днів
    5) Кожного тижня робити:
       - прогноз demand на 7 днів
       - оптимізація замовлень
       - place_order для кожного магазину
       - кожного дня: process_orders і віднімання “продажів” відповідно до прогнозу
       - збір метрик (total_cost, fill_rate)
    6) Зберегти results у CSV (data/processed/simulation_results.csv)
    """

    # Шляхи
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/forecast_model.pkl"))
    store_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/store.csv"))
    initial_stock_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed/initial_stock.csv"))

    # 1) ForecastAgent
    fa = ForecastAgent(model_path=model_path)

    # 2) Початкові запаси
    df_initial = pd.read_csv(initial_stock_csv)
    initial_stock = dict(zip(df_initial["Store"], df_initial["InitialStock"]))

    # 3) InventoryAgent & SupplierAgent
    # Параметри оптимізації (можна коригувати)
    alpha = 5  # вартість дефіциту
    beta = 1   # вартість надлишку
    Q_max = 30000  # max одиниць на тиждень
    ia = InventoryAgent(alpha=alpha, beta=beta, Q_max=Q_max, initial_stock=initial_stock)
    sa = SupplierAgent(delivery_delay_days=2, daily_limit=45000)

    # 4) Параметри симуляції
    current_date = date(2025, 1, 1)
    end_date = date(2025, 3, 31)
    step = timedelta(days=7)

    records = []
    list_of_store_ids = list(initial_stock.keys())

    while current_date <= end_date:
        # 4.1) Збираємо прогноз на 7 днів попиту
        demands = {}
        for store_id in list_of_store_ids:
            preds = fa.predict(store_id=store_id,
                               start_date=pd.Timestamp(current_date),
                               horizon_days=7,
                               store_csv=store_csv)
            demands[store_id] = int(preds.sum())

        # 4.2) Оптимізуємо замовлення
        orders = ia.optimize_orders(demands)

        # 4.3) Place orders (зменшимо тимчасово stock на замовлену кількість, щоб не допустити повторного використання того ж запасу)
        for store_id, qty in orders.items():
            # Не зменшуємо реальний stock тут, бо постачання відбувається через sa.process_orders
            sa.place_order(store_id=store_id, qty=qty, order_date=current_date)

        # 4.4) Протягом кожного дня тижня:
        for day_offset in range(7):
            day = current_date + timedelta(days=day_offset)
            # 4.4.1) Постачальник обробляє чергу
            sa.process_orders(current_date=day, inventory_agent=ia)

            # 4.4.2) Зменшуємо запаси магазину відповідно до “фактичного” продажу
            for store_id in list_of_store_ids:
                # Прогноз на один день вперед від поточної дати
                daily_pred = fa.predict(store_id=store_id,
                                        start_date=pd.Timestamp(day),
                                        horizon_days=1,
                                        store_csv=store_csv)[0]
                ia.stock[store_id] = max(ia.stock[store_id] - int(daily_pred), 0)

        # 4.5) Збираємо метрики на кінець тижня
        total_cost = calculate_total_cost(inventory_agent=ia, demands=demands, alpha=alpha, beta=beta)
        fill_rate = calculate_fill_rate(inventory_agent=ia, demands=demands)
        records.append({
            "week_start": current_date,
            "total_cost": total_cost,
            "fill_rate": fill_rate
        })

        print(f"[Simulation] Week {current_date} → cost={total_cost:.2f}, fill_rate={fill_rate:.3f}")
        current_date += step

    # 4.6) Зберігаємо результати
    df_records = pd.DataFrame(records)
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed/simulation_results.csv"))
    df_records.to_csv(out_path, index=False)
    print(f"[Simulation] Результати симуляції збережено у {out_path}")


if __name__ == "__main__":
    main()

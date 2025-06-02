import streamlit as st
import pandas as pd
import os

# Ми припускаємо, що у PYTHONPATH вже є папка src, або ж запускаємо цей скрипт із кореня проекту.
from src.forecast_agent.forecast_agent import ForecastAgent
from src.inventory_agent.inventory_agent import InventoryAgent
from src.supplier_agent.supplier_agent import SupplierAgent

# Налаштування шляхи
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "models/forecast_model.pkl"))
STORE_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/raw/store.csv"))
INITIAL_STOCK_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/processed/initial_stock.csv"))
SIMULATION_RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/processed/simulation_results.csv"))

# Завантажуємо список магазинів та початкові запаси
df_stock = pd.read_csv(INITIAL_STOCK_CSV)
list_of_store_ids = df_stock["Store"].tolist()
current_stock_dict = dict(zip(df_stock["Store"], df_stock["InitialStock"]))

# Ініціалізуємо ForecastAgent на старті
fa = ForecastAgent(model_path=MODEL_PATH)

st.title("Інтелектуальне управління запасами для роздрібної мережі")

menu = ["Прогнозування", "Оптимізація запасів", "Симуляція"]
choice = st.sidebar.selectbox("Меню", menu)

if choice == "Прогнозування":
    st.header("ForecastAgent: Прогноз продажів")
    store_id = st.selectbox("Виберіть магазин", list_of_store_ids)
    start_date = st.date_input("Дата початку прогнозу", value=pd.to_datetime("2025-01-01"))
    horizon = st.slider("Горизонт прогнозу (днів)", min_value=7, max_value=30, value=14)
    if st.button("Отримати прогноз"):
        preds = fa.predict(store_id=store_id,
                           start_date=pd.Timestamp(start_date),
                           horizon_days=horizon,
                           store_csv=STORE_CSV)
        df_plot = pd.DataFrame({
            "day": pd.date_range(start_date, periods=horizon),
            "predicted_sales": preds
        })
        df_plot = df_plot.set_index("day")
        st.line_chart(df_plot)

elif choice == "Оптимізація запасів":
    st.header("InventoryAgent: Оптимізація замовлення")
    week_start = st.date_input("Дата початку тижня для оптимізації", value=pd.to_datetime("2025-01-01"))
    alpha = st.number_input("Вартість дефіциту (alpha)", value=5.0)
    beta = st.number_input("Вартість надлишку (beta)", value=1.0)
    Q_max = st.number_input("Максимальний обсяг постачання за тиждень (Q_max)", value=30000)
    if st.button("Оптимізувати"):
        demands = {}
        for sid in list_of_store_ids:
            preds = fa.predict(store_id=sid,
                               start_date=pd.Timestamp(week_start),
                               horizon_days=7,
                               store_csv=STORE_CSV)
            demands[sid] = int(preds.sum())
        ia = InventoryAgent(alpha=alpha, beta=beta, Q_max=int(Q_max), initial_stock=current_stock_dict)
        orders = ia.optimize_orders(demands)
        df_orders = pd.DataFrame.from_dict(orders, orient="index", columns=["qty_to_order"])
        st.dataframe(df_orders)

elif choice == "Симуляція":
    st.header("Повна симуляція (90 днів)")
    if st.button("Запустити симуляцію"):
        # Викликаємо main() із simulation.py
        from src.simulation.simulation import main as run_simulation
        run_simulation()
        st.success("Симуляція завершена! Результати збережено.")

    # Якщо результати вже є, відобразимо їх
    if os.path.exists(SIMULATION_RESULTS):
        df_res = pd.read_csv(SIMULATION_RESULTS, parse_dates=["week_start"])
        df_res = df_res.sort_values("week_start")
        st.subheader("Сума витрат по тижнях")
        st.line_chart(df_res.set_index("week_start")["total_cost"])
        st.subheader("Fill Rate по тижнях")
        st.line_chart(df_res.set_index("week_start")["fill_rate"])

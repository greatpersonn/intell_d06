import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_simulation_results():
    """
    1) Зчитати data/processed/simulation_results.csv
    2) Побудувати два графіки: total_cost vs week_start, fill_rate vs week_start
    3) Зберегти зображення або просто відобразити через matplotlib
    """
    results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed/simulation_results.csv"))
    df = pd.read_csv(results_path, parse_dates=["week_start"])
    df = df.sort_values("week_start")

    # 1) Графік total_cost
    plt.figure(figsize=(8, 4))
    plt.plot(df["week_start"], df["total_cost"], marker="o")
    plt.title("Total Cost per Week")
    plt.xlabel("Week Start")
    plt.ylabel("Total Cost")
    plt.tight_layout()
    plt.savefig("total_cost.png")
    plt.close()

    # 2) Графік fill_rate
    plt.figure(figsize=(8, 4))
    plt.plot(df["week_start"], df["fill_rate"], marker="o", color="green")
    plt.title("Fill Rate per Week")
    plt.xlabel("Week Start")
    plt.ylabel("Fill Rate")
    plt.tight_layout()
    plt.savefig("fill_rate.png")
    plt.close()

    print("[plot_results] Графіки збережено: total_cost.png, fill_rate.png")

if __name__ == "__main__":
    plot_simulation_results()

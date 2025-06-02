def calculate_total_cost(inventory_agent, demands: dict, alpha: float, beta: float) -> float:
    """
    Обчислює загальні витрати (дефіцит + надлишок) для поточного stock і заданих demands.
    demands: {store_id: demand_for_week}
    alpha, beta — такі ж, як у InventoryAgent
    """
    total_cost = 0.0
    for store_id, p_i in demands.items():
        s_i = inventory_agent.stock.get(store_id, 0)
        # Якщо товар вже доставлено, stock включає новий запас
        # Вважаємо, що q_i вже був доданий у stock на момент виклику
        # Дефіцит:
        deficit = max(p_i - s_i, 0)
        over = max(s_i - p_i, 0)
        total_cost += alpha * deficit + beta * over
    return total_cost

def calculate_fill_rate(inventory_agent, demands: dict) -> float:
    """
    Обчислює fill rate: (сума задоволеного попиту / сума прогнозного попиту)
    Припускаємо, що якщо stock >= demand, то весь попит задовольняється; 
    якщо stock < demand, то задовольняється лише stock.
    """
    total_demand = 0.0
    total_fulfilled = 0.0
    for store_id, p_i in demands.items():
        s_i = inventory_agent.stock.get(store_id, 0)
        fulfilled = min(s_i, p_i)
        total_demand += p_i
        total_fulfilled += fulfilled
    if total_demand == 0:
        return 1.0
    return total_fulfilled / total_demand

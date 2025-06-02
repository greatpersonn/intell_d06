from datetime import timedelta

class SupplierAgent:
    """
    Імітує постачання з фіксованою затримкою та лімітом на добу.
    Аргументи:
        delivery_delay_days: кількість днів затримки (наприклад, 2)
        daily_limit: максимально можна відвантажити в один день (сума по всіх замовленнях)
    """

    def __init__(self, delivery_delay_days: int = 2, daily_limit: int = 45000):
        self.delivery_delay = timedelta(days=delivery_delay_days)
        self.daily_limit = daily_limit
        # order_queue: список словників:
        # [{"store_id": int, "order_qty": int, "order_date": datetime.date, "remaining_qty": int, "delivered": bool}, ...]
        self.order_queue = []

    def place_order(self, store_id: int, qty: int, order_date):
        """
        Додати замовлення в чергу.
        """
        self.order_queue.append({
            "store_id": store_id,
            "order_qty": qty,
            "order_date": order_date,
            "remaining_qty": qty,
            "delivered": False
        })

    def process_orders(self, current_date, inventory_agent):
        """
        Щодня викликається з поточною датою:
        - Перевіряє, чи настав час доставки (order_date + delay <= current_date)
        - Якщо можна доставляти і ліміт не вичерпано, додає товари до inventory_agent.stock
        - Якщо ліміт вичерпано, лишок чекає наступного дня
        """
        deliverable = self.daily_limit
        for order in self.order_queue:
            if order["delivered"]:
                continue

            if order["order_date"] + self.delivery_delay <= current_date:
                # можемо доставляти
                to_send = min(order["remaining_qty"], deliverable)
                if to_send > 0:
                    # Доставляємо
                    inventory_agent.stock[order["store_id"]] += to_send
                    order["remaining_qty"] -= to_send
                    deliverable -= to_send
                    if order["remaining_qty"] == 0:
                        order["delivered"] = True
                # Якщо daily_limit вичерпано, лишок чекатиме наступного дня
        # За бажанням можна чистити повністю доставлені замовлення:
        # self.order_queue = [o for o in self.order_queue if not o["delivered"]]

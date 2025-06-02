import random
from deap import base, creator, tools, algorithms

class InventoryAgent:
    """
    InventoryAgent реалізує оптимізацію замовлень за допомогою генетичного алгоритму (DEAP).
    Аргументи:
        alpha: вартість дефіциту за одиницю
        beta: вартість надлишкового зберігання за одиницю
        Q_max: максимально доступна кількість одиниць для відвантаження (на період)
        initial_stock: словник {store_id: поточний_stock}
    """

    def __init__(self, alpha: float, beta: float, Q_max: int, initial_stock: dict):
        self.alpha = alpha
        self.beta = beta
        self.Q_max = Q_max
        self.stock = initial_stock.copy()  # {store_id: units}
        self.store_ids = list(self.stock.keys())

        # DEAP setup (створимо класи, якщо ще не створені)
        # Уникаємо дублювання creator: перевіряємо, чи вже є такі класи
        try:
            creator.FitnessMax
        except AttributeError:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        try:
            creator.Individual
        except AttributeError:
            creator.create("Individual", list, fitness=creator.FitnessMax)

    def fitness(self, individual, demands: dict):
        """
        Функція пристосованості (fitness): 
        - individual: список [q1, q2, ..., qN] замовлень для кожного магазину
        - demands: {store_id: demand_for_week}
        Повертає кортеж ( —cost ), оскільки ми хочемо мінімізувати cost → максимізуємо -cost
        """
        cost = 0
        for idx, store_id in enumerate(self.store_ids):
            p_i = demands.get(store_id, 0)
            s_i = self.stock.get(store_id, 0)
            q_i = individual[idx]
            # дефіцит
            deficit = max(p_i - (s_i + q_i), 0)
            # надлишок
            over = max((s_i + q_i) - p_i, 0)
            cost += self.alpha * deficit + self.beta * over

        # штраф за перевищення Q_max
        total_order = sum(individual)
        if total_order > self.Q_max:
            cost += (total_order - self.Q_max) * 1000  # великий штраф

        return (-cost,)

    def optimize_orders(self, demands: dict) -> dict:
        """
        demands: {store_id: demand_for_week}
        Повертає словник {store_id: optimal_q}.
        Після оптимізації не змінює self.stock — доставка моделюється окремо через SupplierAgent.
        """
        N = len(self.store_ids)

        # 1) Налаштуємо toolbox
        toolbox = base.Toolbox()
        toolbox.register("attr_int", random.randint, 0, self.Q_max)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=N)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.fitness, demands=demands)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=self.Q_max, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # 2) Ініціалізуємо популяцію
        pop = toolbox.population(n=50)

        # 3) Еволюція
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda x: sum(f[0] for f in x) / len(x))
        stats.register("max", max)

        pop, logbook = algorithms.eaSimple(pop, toolbox,
                                           cxpb=0.5, mutpb=0.2,
                                           ngen=40, stats=stats, verbose=False)
        # 4) Найкращий індивід
        best_ind = tools.selBest(pop, 1)[0]
        orders = { self.store_ids[i]: best_ind[i] for i in range(N) }
        return orders

from functools import lru_cache
from typing import List, Tuple, Dict, Any
from flask import Flask, request, jsonify

app = Flask(__name__)

class RidePricingOptimizer:
    def __init__(self) -> None:
        self.cost_per_ride: float = 3.0
        self.price_options: List[int] = [8, 22]  # Available price points

    def optimize_prices(self, base_demands: List[int]) -> List[int]:
        """
        Optimize ride prices using DP.
        """
        T: int = len(base_demands)
        price_options: List[int] = self.price_options
        cost: float = self.cost_per_ride

        @lru_cache(maxsize=None)
        def dp(t: int, deferred: int) -> Tuple[float, List[int]]:
            if t == T:
                return 0.0, []
            best_profit: float = 0
            best_path: List[int] = []

            for p in price_options:
                total_demand: int = base_demands[t] + deferred
                actual_customers, new_deferred = self.calculate_demand_and_spillover(
                    total_demand, p
                )
                profit: float = (p - cost) * actual_customers

                future_profit, path = dp(t + 1, new_deferred)
                total_profit: float = profit + future_profit

                if total_profit > best_profit:
                    best_profit = total_profit
                    best_path = [p] + path

            return best_profit, best_path
        profit, prices = dp(0, 0)
        return prices
    def brute_force_prices(self, base_demands: List[int]) -> List[int]:
        """
        Brute-force DFS:
        Try all possible price combinations.
        """
        T = len(base_demands)
        best_profit = float("-inf")
        best_path: List[int] = []

        def dfs(t: int, deferred: int, profit_so_far: float, path: List[int]):
            nonlocal best_profit, best_path
            if t == T:
                if profit_so_far > best_profit:
                    best_profit = profit_so_far
                    best_path = path[:]
                return

            total_demand = base_demands[t] + deferred
            for p in self.price_options:
                actual_customers, new_deferred = self.calculate_demand_and_spillover(
                    total_demand, p
                )
                period_profit = (p - self.cost_per_ride) * actual_customers
                dfs(
                    t + 1,
                    new_deferred,
                    profit_so_far + period_profit,
                    path + [p],
                )

        dfs(0, 0, 0.0, [])
        return best_path
    def compare_algorithms(self, base_demands: List[int]) -> Dict[str, Dict]:
        """
        Compare DP vs Brute-force pricing strategies.
        """
        results: Dict[str, Dict] = {}

        # DP
        dp_prices = self.optimize_prices(base_demands)
        dp_profit = self.simulate_day(base_demands, dp_prices)
        results["Dynamic Programming"] = {
            "prices": dp_prices,
            "profit": dp_profit,
        }

        # Brute Force
        bf_prices = self.brute_force_prices(base_demands)
        bf_profit = self.simulate_day(base_demands, bf_prices)
        results["Brute Force"] = {
            "prices": bf_prices,
            "profit": bf_profit,
        }

        return results

    def simulate_day(self, base_demands: List[int], prices: List[int]) -> float:
        """
        Simulate one full day with the given prices and return total profit.

        Args:
            base_demands: Base demand for each period
            prices: Price in each period

        Returns:
            Total profit for the day
        """
        if len(base_demands) != len(prices):
            raise ValueError("base_demands and prices must have same length")

        total_profit: float = 0.0
        deferred_customers: int = 0  # Number of spillover customers

        for period in range(len(base_demands)):
            # Total demand = base demand + spillover
            total_demand: int = base_demands[period] + deferred_customers

            actual_customers, new_deferred = self.calculate_demand_and_spillover(
                total_demand, prices[period]
            )

            # Calculate profit for this period
            period_profit: float = (prices[period] - self.cost_per_ride) * actual_customers
            total_profit += period_profit

            # Update spillover customers for next period
            deferred_customers = new_deferred

        return total_profit

    def calculate_demand_and_spillover(self, total_demand: int, price: int) -> Tuple[int, int]:
        """
        Calculate actual ridership and spillover based on price.

        Args:
            total_demand: Total customers wanting rides this period
            price: Price being charged

        Returns:
            (actual_customers_this_period, customers_deferred_to_next_period)
        """
        if price <= 10:
            return total_demand, 0
        elif price <= 20:
            deferred: int = round(total_demand * 0.1)
            actual: int = total_demand - deferred
            return actual, deferred
        else:  # price >= 21
            deferred: int = round(total_demand * 0.3)
            actual: int = total_demand - deferred
            return actual, deferred


optimizer = RidePricingOptimizer()

@app.route("/optimize-prices", methods=["POST"])
def create_pricing_api_endpoint() -> Any:
    """Register Flask route"""
    data: Dict[str, Any] = request.get_json(force=True)
    if not data or "base_demands" not in data:
        return jsonify({"Error": "Base_demands not provided"}), 400

    base_demands: List[int] = data["base_demands"]
    prices: List[int] = optimizer.optimize_prices(base_demands)
    profit: float = optimizer.simulate_day(base_demands, prices)

    return jsonify({
        "base_demands": base_demands,
        "optimal_prices": prices,
        "expected_profit": profit
    })


if __name__ == "__main__":
    import sys

    optimizer = RidePricingOptimizer()

    if "api" in sys.argv:
        # Run the Flask app, if specified in terminal
        app.run(debug=True)
    else:
        # Sample demand pattern: morning, lunch, evening, night
        base_demands = [30, 20, 50, 25]  # Small example for testing

        print(f"Base demands: {base_demands}")
        print(f"Available prices: {optimizer.price_options}")
        print(f"Cost per ride: ${optimizer.cost_per_ride}\n")

        # Test simulation
        test_prices = [8, 10, 12, 8]
        test_profit = optimizer.simulate_day(base_demands, test_prices)
        print(f"Test prices {test_prices} -> Profit: ${test_profit:.2f}\n")

        # Run your optimization
        results = optimizer.compare_algorithms(base_demands)

        for algorithm, result in results.items():
            print(f"{algorithm}:")
            print(f"  Prices: {result['prices']}")
            print(f"  Profit: ${result['profit']:.2f}")
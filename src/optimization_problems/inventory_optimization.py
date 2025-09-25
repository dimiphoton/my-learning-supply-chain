"""
Supply Chain Optimization Problems

This module contains practical optimization problems commonly found in
supply chain management. It includes both classical problems and modern
data-driven approaches using Python optimization libraries.

Key Problems Covered:
1. Economic Order Quantity (EOQ)
2. Transportation Problem
3. Facility Location Problem
4. Inventory Optimization with Uncertainty
5. Multi-objective Optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    warnings.warn("PuLP not available. Install with: pip install pulp")

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Install with: pip install scipy")


class EOQOptimizer:
    """Economic Order Quantity optimization with extensions."""
    
    def __init__(self, demand: float, ordering_cost: float, holding_cost: float):
        """
        Initialize EOQ optimizer.
        
        Args:
            demand: Annual demand
            ordering_cost: Cost per order
            holding_cost: Holding cost per unit per year
        """
        self.demand = demand
        self.ordering_cost = ordering_cost
        self.holding_cost = holding_cost
    
    def basic_eoq(self) -> Dict:
        """Calculate basic Economic Order Quantity."""
        
        # EOQ formula: sqrt(2 * D * S / H)
        eoq = np.sqrt(2 * self.demand * self.ordering_cost / self.holding_cost)
        
        # Calculate associated costs
        total_ordering_cost = (self.demand / eoq) * self.ordering_cost
        total_holding_cost = (eoq / 2) * self.holding_cost
        total_cost = total_ordering_cost + total_holding_cost
        
        # Calculate number of orders per year
        orders_per_year = self.demand / eoq
        
        # Calculate cycle length
        cycle_length = eoq / self.demand * 365  # in days
        
        return {
            'eoq': round(eoq, 2),
            'total_cost': round(total_cost, 2),
            'ordering_cost': round(total_ordering_cost, 2),
            'holding_cost': round(total_holding_cost, 2),
            'orders_per_year': round(orders_per_year, 2),
            'cycle_length_days': round(cycle_length, 1)
        }
    
    def eoq_with_quantity_discount(self, discount_brackets: List[Tuple]) -> Dict:
        """
        Calculate EOQ with quantity discounts.
        
        Args:
            discount_brackets: List of (quantity, unit_cost) tuples
        
        Returns:
            Dict with optimal order quantity and cost breakdown
        """
        results = []
        
        for min_qty, unit_cost in discount_brackets:
            # Calculate EOQ for this price level
            eoq_result = self.basic_eoq()
            eoq_qty = eoq_result['eoq']
            
            # Check if EOQ is feasible for this bracket
            if eoq_qty >= min_qty:
                order_qty = eoq_qty
            else:
                order_qty = min_qty
            
            # Calculate total cost including purchase cost
            annual_purchase_cost = self.demand * unit_cost
            ordering_cost = (self.demand / order_qty) * self.ordering_cost
            holding_cost = (order_qty / 2) * self.holding_cost
            
            total_annual_cost = annual_purchase_cost + ordering_cost + holding_cost
            
            results.append({
                'min_quantity': min_qty,
                'unit_cost': unit_cost,
                'order_quantity': round(order_qty, 2),
                'purchase_cost': round(annual_purchase_cost, 2),
                'ordering_cost': round(ordering_cost, 2),
                'holding_cost': round(holding_cost, 2),
                'total_cost': round(total_annual_cost, 2)
            })
        
        # Find optimal solution
        optimal = min(results, key=lambda x: x['total_cost'])
        
        return {
            'all_options': results,
            'optimal_solution': optimal,
            'savings_vs_basic_eoq': round(
                self.basic_eoq()['total_cost'] + self.demand * discount_brackets[0][1] - optimal['total_cost'], 2
            )
        }
    
    def plot_eoq_analysis(self, figsize: tuple = (12, 8)):
        """Plot EOQ cost analysis."""
        
        eoq_result = self.basic_eoq()
        optimal_qty = eoq_result['eoq']
        
        # Create range of order quantities around EOQ
        quantities = np.linspace(optimal_qty * 0.3, optimal_qty * 2, 100)
        
        ordering_costs = (self.demand / quantities) * self.ordering_cost
        holding_costs = (quantities / 2) * self.holding_cost
        total_costs = ordering_costs + holding_costs
        
        plt.figure(figsize=figsize)
        
        plt.plot(quantities, ordering_costs, label='Ordering Cost', linestyle='--', alpha=0.7)
        plt.plot(quantities, holding_costs, label='Holding Cost', linestyle='--', alpha=0.7)
        plt.plot(quantities, total_costs, label='Total Cost', linewidth=3)
        
        # Mark optimal point
        plt.axvline(x=optimal_qty, color='red', linestyle=':', alpha=0.8, label=f'EOQ = {optimal_qty:.0f}')
        plt.plot(optimal_qty, eoq_result['total_cost'], 'ro', markersize=10, label=f'Minimum Cost = ${eoq_result["total_cost"]:,.0f}')
        
        plt.xlabel('Order Quantity')
        plt.ylabel('Annual Cost ($)')
        plt.title('Economic Order Quantity (EOQ) Analysis', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class TransportationOptimizer:
    """Transportation problem optimizer using linear programming."""
    
    def __init__(self, supply: List[float], demand: List[float], costs: np.ndarray):
        """
        Initialize transportation problem.
        
        Args:
            supply: List of supply capacities
            demand: List of demand requirements
            costs: Cost matrix (suppliers x customers)
        """
        self.supply = supply
        self.demand = demand
        self.costs = np.array(costs)
        
        # Check if problem is balanced
        self.is_balanced = sum(supply) == sum(demand)
        
        if not self.is_balanced:
            print(f"Warning: Unbalanced problem. Total supply: {sum(supply)}, Total demand: {sum(demand)}")
    
    def solve_transportation_problem(self) -> Dict:
        """Solve transportation problem using linear programming."""
        
        if not PULP_AVAILABLE:
            raise ImportError("PuLP is required for this optimization. Install with: pip install pulp")
        
        m, n = self.costs.shape  # m suppliers, n customers
        
        # Create the problem
        prob = pulp.LpProblem("Transportation_Problem", pulp.LpMinimize)
        
        # Decision variables
        x = {}
        for i in range(m):
            for j in range(n):
                x[i, j] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0)
        
        # Objective function: minimize total transportation cost
        prob += pulp.lpSum([self.costs[i][j] * x[i, j] for i in range(m) for j in range(n)])
        
        # Supply constraints
        for i in range(m):
            prob += pulp.lpSum([x[i, j] for j in range(n)]) <= self.supply[i], f"Supply_{i}"
        
        # Demand constraints
        for j in range(n):
            prob += pulp.lpSum([x[i, j] for i in range(m)]) >= self.demand[j], f"Demand_{j}"
        
        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract solution
        if prob.status == pulp.LpStatusOptimal:
            solution = np.zeros((m, n))
            for i in range(m):
                for j in range(n):
                    solution[i, j] = x[i, j].varValue
            
            total_cost = pulp.value(prob.objective)
            
            return {
                'status': 'Optimal',
                'total_cost': round(total_cost, 2),
                'solution_matrix': solution,
                'utilization': self._calculate_utilization(solution)
            }
        else:
            return {'status': 'No optimal solution found'}
    
    def _calculate_utilization(self, solution: np.ndarray) -> Dict:
        """Calculate supply and demand utilization."""
        
        supply_used = solution.sum(axis=1)
        demand_met = solution.sum(axis=0)
        
        supply_utilization = (supply_used / self.supply * 100).round(1)
        demand_satisfaction = (demand_met / self.demand * 100).round(1)
        
        return {
            'supply_utilization': supply_utilization.tolist(),
            'demand_satisfaction': demand_satisfaction.tolist()
        }
    
    def create_solution_summary(self, solution_result: Dict) -> pd.DataFrame:
        """Create a readable summary of the transportation solution."""
        
        if solution_result['status'] != 'Optimal':
            return pd.DataFrame({'Message': ['No optimal solution available']})
        
        solution = solution_result['solution_matrix']
        m, n = solution.shape
        
        summary_data = []
        for i in range(m):
            for j in range(n):
                if solution[i, j] > 0:  # Only include non-zero shipments
                    summary_data.append({
                        'From': f'Supplier_{i+1}',
                        'To': f'Customer_{j+1}',
                        'Quantity': round(solution[i, j], 2),
                        'Unit_Cost': self.costs[i, j],
                        'Total_Cost': round(solution[i, j] * self.costs[i, j], 2)
                    })
        
        df = pd.DataFrame(summary_data)
        if not df.empty:
            df = df.sort_values(['From', 'To']).reset_index(drop=True)
        
        return df


class InventoryOptimizer:
    """Multi-SKU inventory optimization with uncertainty."""
    
    def __init__(self, sku_data: pd.DataFrame):
        """
        Initialize with SKU data.
        
        Expected columns: demand_mean, demand_std, unit_cost, holding_rate, 
                         ordering_cost, service_level_target
        """
        self.data = sku_data.copy()
        
    def optimize_safety_stock(self) -> pd.DataFrame:
        """Optimize safety stock levels based on service level targets."""
        
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required. Install with: pip install scipy")
            
        from scipy.stats import norm
        
        results = self.data.copy()
        
        # Calculate safety stock for each SKU
        z_scores = norm.ppf(results['service_level_target'])
        results['safety_stock'] = z_scores * results['demand_std']
        results['safety_stock'] = results['safety_stock'].clip(lower=0)  # Non-negative
        
        # Calculate EOQ for each SKU
        annual_demand = results['demand_mean'] * 52  # Weekly to annual
        holding_cost_per_unit = results['unit_cost'] * results['holding_rate']
        
        results['eoq'] = np.sqrt(2 * annual_demand * results['ordering_cost'] / holding_cost_per_unit)
        
        # Calculate total inventory investment
        results['cycle_stock'] = results['eoq'] / 2
        results['total_inventory'] = results['cycle_stock'] + results['safety_stock']
        results['inventory_value'] = results['total_inventory'] * results['unit_cost']
        
        # Calculate expected stockout frequency
        results['expected_stockouts_per_year'] = 52 * (1 - results['service_level_target'])
        
        return results.round(2)
    
    def trade_off_analysis(self, service_levels: List[float]) -> pd.DataFrame:
        """Analyze trade-off between service level and inventory investment."""
        
        trade_off_results = []
        
        for service_level in service_levels:
            # Temporarily set all SKUs to this service level
            temp_data = self.data.copy()
            temp_data['service_level_target'] = service_level
            
            temp_optimizer = InventoryOptimizer(temp_data)
            optimized = temp_optimizer.optimize_safety_stock()
            
            total_inventory_value = optimized['inventory_value'].sum()
            total_stockouts = optimized['expected_stockouts_per_year'].sum()
            
            trade_off_results.append({
                'service_level': service_level * 100,  # Convert to percentage
                'total_inventory_value': round(total_inventory_value, 0),
                'expected_stockouts_per_year': round(total_stockouts, 1),
                'average_safety_stock': round(optimized['safety_stock'].mean(), 2)
            })
        
        return pd.DataFrame(trade_off_results)


def create_sample_optimization_problems():
    """Create sample problems for demonstration."""
    
    # EOQ Example
    eoq_example = {
        'demand': 10000,  # units per year
        'ordering_cost': 50,  # $ per order
        'holding_cost': 2.5,  # $ per unit per year
        'description': 'Electronic component with steady demand'
    }
    
    # Transportation Problem Example
    transportation_example = {
        'supply': [400, 300, 500],  # 3 suppliers
        'demand': [250, 350, 400, 200],  # 4 customers
        'costs': [
            [8, 6, 10, 9],   # Costs from supplier 1 to customers 1-4
            [9, 12, 13, 7],  # Costs from supplier 2 to customers 1-4
            [14, 9, 16, 5]   # Costs from supplier 3 to customers 1-4
        ],
        'description': 'Distribution network optimization'
    }
    
    # Multi-SKU Inventory Example
    np.random.seed(42)
    n_skus = 10
    
    inventory_example = pd.DataFrame({
        'sku_id': [f'SKU_{i:03d}' for i in range(1, n_skus + 1)],
        'demand_mean': np.random.uniform(50, 500, n_skus).round(0),
        'demand_std': np.random.uniform(5, 50, n_skus).round(1),
        'unit_cost': np.random.uniform(10, 100, n_skus).round(2),
        'holding_rate': np.random.uniform(0.15, 0.25, n_skus).round(3),
        'ordering_cost': np.random.uniform(25, 75, n_skus).round(0),
        'service_level_target': np.random.uniform(0.90, 0.99, n_skus).round(2)
    })
    
    return {
        'eoq_example': eoq_example,
        'transportation_example': transportation_example,
        'inventory_example': inventory_example
    }


def demonstrate_optimization_problems():
    """Demonstrate various supply chain optimization problems."""
    
    print("=== Supply Chain Optimization Problems ===\n")
    
    # Get sample problems
    examples = create_sample_optimization_problems()
    
    # 1. EOQ Demonstration
    print("1. Economic Order Quantity (EOQ) Optimization")
    print(f"   Problem: {examples['eoq_example']['description']}")
    
    eoq_opt = EOQOptimizer(**{k: v for k, v in examples['eoq_example'].items() if k != 'description'})
    eoq_result = eoq_opt.basic_eoq()
    
    print(f"   Annual Demand: {examples['eoq_example']['demand']:,} units")
    print(f"   Ordering Cost: ${examples['eoq_example']['ordering_cost']}")
    print(f"   Holding Cost: ${examples['eoq_example']['holding_cost']} per unit per year")
    print(f"\n   📊 Optimal Solution:")
    print(f"   - Order Quantity: {eoq_result['eoq']} units")
    print(f"   - Total Annual Cost: ${eoq_result['total_cost']:,.2f}")
    print(f"   - Orders per Year: {eoq_result['orders_per_year']}")
    print(f"   - Cycle Length: {eoq_result['cycle_length_days']} days")
    
    # 2. Transportation Problem
    if PULP_AVAILABLE:
        print(f"\n2. Transportation Problem Optimization")
        print(f"   Problem: {examples['transportation_example']['description']}")
        
        transport_data = examples['transportation_example']
        transport_opt = TransportationOptimizer(
            transport_data['supply'], 
            transport_data['demand'], 
            transport_data['costs']
        )
        transport_result = transport_opt.solve_transportation_problem()
        
        if transport_result['status'] == 'Optimal':
            print(f"   📊 Optimal Solution:")
            print(f"   - Total Transportation Cost: ${transport_result['total_cost']:,.2f}")
            print(f"   - Number of Active Routes: {np.count_nonzero(transport_result['solution_matrix'])}")
            
            # Show solution summary
            summary_df = transport_opt.create_solution_summary(transport_result)
            if not summary_df.empty:
                print(f"\n   Top 3 Shipment Routes:")
                top_routes = summary_df.nlargest(3, 'Quantity')[['From', 'To', 'Quantity', 'Total_Cost']]
                for _, route in top_routes.iterrows():
                    print(f"   - {route['From']} → {route['To']}: {route['Quantity']} units (${route['Total_Cost']})")
        else:
            print(f"   ❌ Could not find optimal solution")
    else:
        print(f"\n2. Transportation Problem Optimization")
        print("   ⚠️  Requires PuLP library (pip install pulp)")
    
    # 3. Multi-SKU Inventory Optimization
    if SCIPY_AVAILABLE:
        print(f"\n3. Multi-SKU Inventory Optimization")
        print("   Problem: Optimizing safety stock for multiple products")
        
        inventory_opt = InventoryOptimizer(examples['inventory_example'])
        inventory_result = inventory_opt.optimize_safety_stock()
        
        total_investment = inventory_result['inventory_value'].sum()
        avg_service_level = inventory_result['service_level_target'].mean()
        
        print(f"   📊 Optimization Results:")
        print(f"   - Total Inventory Investment: ${total_investment:,.0f}")
        print(f"   - Average Service Level: {avg_service_level:.1%}")
        print(f"   - Number of SKUs: {len(inventory_result)}")
        
        # Show top 3 SKUs by inventory value
        top_skus = inventory_result.nlargest(3, 'inventory_value')[['sku_id', 'total_inventory', 'inventory_value']]
        print(f"\n   Top 3 SKUs by Inventory Value:")
        for _, sku in top_skus.iterrows():
            print(f"   - {sku['sku_id']}: {sku['total_inventory']:.0f} units (${sku['inventory_value']:,.0f})")
    else:
        print(f"\n3. Multi-SKU Inventory Optimization")
        print("   ⚠️  Requires SciPy library (pip install scipy)")
    
    print(f"\n4. Key Optimization Principles in Supply Chain:")
    print("   🎯 Trade-offs: Cost vs Service vs Speed vs Quality")
    print("   📈 Constraints: Capacity, Budget, Time, Regulations")
    print("   🔄 Dynamics: Demand uncertainty, lead time variability")
    print("   📊 Multi-objective: Balancing conflicting objectives")
    
    return examples


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_optimization_problems()
    
    print("\n" + "="*60)
    print("🚀 Next Steps:")
    print("   - Try modifying the parameters in the examples")
    print("   - Apply these techniques to your own supply chain data")
    print("   - Explore advanced topics like stochastic optimization")
    print("   - Check out the Jupyter notebooks for interactive examples")
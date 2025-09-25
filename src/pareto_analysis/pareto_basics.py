"""
Pareto Analysis (80/20 Principle) in Supply Chain

This module demonstrates how to apply the Pareto principle to various
supply chain scenarios. The 80/20 rule states that roughly 80% of effects
come from 20% of causes.

In supply chain context:
- 80% of revenue typically comes from 20% of products
- 80% of inventory costs from 20% of SKUs
- 80% of quality issues from 20% of suppliers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple


class ParetoAnalyzer:
    """Class to perform Pareto analysis on supply chain data."""
    
    def __init__(self):
        self.data = None
        self.analysis_results = {}
    
    def load_data(self, data: pd.DataFrame):
        """Load data for analysis."""
        self.data = data.copy()
        return self
    
    def abc_analysis(self, value_column: str, item_column: str = None) -> pd.DataFrame:
        """
        Perform ABC analysis (Pareto-based classification).
        
        Args:
            value_column: Column name containing values (e.g., revenue, cost)
            item_column: Column name containing item identifiers
        
        Returns:
            DataFrame with ABC classification
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        df = self.data.copy()
        if item_column is None:
            item_column = df.index.name if df.index.name else 'Item'
            df = df.reset_index()
        
        # Sort by value in descending order
        df_sorted = df.sort_values(value_column, ascending=False).reset_index(drop=True)
        
        # Calculate cumulative percentage
        df_sorted['cumulative_value'] = df_sorted[value_column].cumsum()
        total_value = df_sorted[value_column].sum()
        df_sorted['cumulative_percent'] = (df_sorted['cumulative_value'] / total_value) * 100
        df_sorted['percent_of_items'] = ((df_sorted.index + 1) / len(df_sorted)) * 100
        
        # ABC Classification
        def classify_abc(cum_percent, item_percent):
            if cum_percent <= 80 and item_percent <= 20:
                return 'A'
            elif cum_percent <= 95:
                return 'B'
            else:
                return 'C'
        
        df_sorted['abc_class'] = df_sorted.apply(
            lambda row: classify_abc(row['cumulative_percent'], row['percent_of_items']), 
            axis=1
        )
        
        # Calculate statistics
        abc_stats = df_sorted.groupby('abc_class').agg({
            value_column: ['count', 'sum'],
            'percent_of_items': 'max'
        }).round(2)
        
        abc_stats.columns = ['item_count', 'total_value', 'max_item_percent']
        abc_stats['value_percent'] = (abc_stats['total_value'] / total_value * 100).round(2)
        
        self.analysis_results['abc_analysis'] = {
            'data': df_sorted,
            'stats': abc_stats
        }
        
        return df_sorted
    
    def pareto_chart(self, value_column: str, item_column: str = None, 
                     title: str = "Pareto Analysis", figsize: tuple = (12, 8)):
        """
        Create a Pareto chart visualization.
        
        Args:
            value_column: Column with values
            item_column: Column with item names
            title: Chart title
            figsize: Figure size tuple
        """
        if 'abc_analysis' not in self.analysis_results:
            self.abc_analysis(value_column, item_column)
        
        df_sorted = self.analysis_results['abc_analysis']['data']
        
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Bar chart for individual values
        bars = ax1.bar(range(len(df_sorted)), df_sorted[value_column], 
                       alpha=0.7, color='steelblue', label=f'{value_column}')
        ax1.set_xlabel('Items (sorted by value)')
        ax1.set_ylabel(f'{value_column}', color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        
        # Line chart for cumulative percentage
        ax2 = ax1.twinx()
        line = ax2.plot(range(len(df_sorted)), df_sorted['cumulative_percent'], 
                        color='red', marker='o', markersize=2, linewidth=2, 
                        label='Cumulative %')
        ax2.set_ylabel('Cumulative Percentage', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 100)
        
        # Add 80% line
        ax2.axhline(y=80, color='green', linestyle='--', alpha=0.7, 
                   label='80% line')
        
        # Add 20% of items line (vertical)
        twenty_percent_items = len(df_sorted) * 0.2
        ax1.axvline(x=twenty_percent_items, color='green', linestyle='--', 
                   alpha=0.7, label='20% of items')
        
        plt.title(title, fontsize=16, fontweight='bold')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        plt.tight_layout()
        plt.show()
    
    def get_pareto_summary(self) -> Dict:
        """Get summary of Pareto analysis results."""
        if 'abc_analysis' not in self.analysis_results:
            raise ValueError("Run abc_analysis() first.")
        
        stats = self.analysis_results['abc_analysis']['stats']
        
        # Focus on A category (typically the "20%" that drives "80%" of value)
        a_stats = stats.loc['A']
        
        return {
            'a_category_items': int(a_stats['item_count']),
            'a_category_value_percent': a_stats['value_percent'],
            'a_category_item_percent': a_stats['max_item_percent'],
            'pareto_ratio': f"{a_stats['max_item_percent']:.0f}/{a_stats['value_percent']:.0f}",
            'interpretation': f"{a_stats['max_item_percent']:.0f}% of items contribute {a_stats['value_percent']:.0f}% of total value"
        }


def create_sample_supply_chain_data():
    """Create sample data for demonstrating Pareto analysis."""
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Sample product data
    products = [f"Product_{i:03d}" for i in range(1, 101)]
    
    # Create realistic revenue distribution (follows power law / Pareto distribution)
    # Few products have high revenue, many have low revenue
    revenues = np.random.pareto(0.5, 100) * 10000
    revenues = np.sort(revenues)[::-1]  # Sort descending
    
    # Create costs (roughly 60-80% of revenue with some variation)
    cost_ratios = np.random.uniform(0.6, 0.8, 100)
    costs = revenues * cost_ratios
    
    # Create supplier data
    suppliers = [f"Supplier_{chr(65 + i % 26)}" for i in range(100)]
    
    # Quality issues (some suppliers have more issues)
    quality_issues = np.random.poisson(2, 100)  # Average 2 issues per supplier
    # Make some suppliers have significantly more issues
    high_issue_suppliers = np.random.choice(100, 20, replace=False)
    quality_issues[high_issue_suppliers] += np.random.poisson(10, 20)
    
    return {
        'product_data': pd.DataFrame({
            'product_id': products,
            'annual_revenue': revenues.round(2),
            'annual_cost': costs.round(2),
            'profit': (revenues - costs).round(2)
        }),
        
        'supplier_data': pd.DataFrame({
            'supplier_id': suppliers,
            'annual_spend': np.random.uniform(50000, 500000, 100).round(2),
            'quality_issues': quality_issues,
            'delivery_delays': np.random.poisson(5, 100)
        })
    }


def demonstrate_pareto_applications():
    """Demonstrate various applications of Pareto analysis in supply chain."""
    
    print("=== Pareto Analysis (80/20 Rule) in Supply Chain ===\n")
    
    # Create sample data
    sample_data = create_sample_supply_chain_data()
    
    # 1. Product Revenue Analysis
    print("1. Product Revenue Analysis (ABC Classification)")
    print("   Identifying which products drive most revenue\n")
    
    analyzer = ParetoAnalyzer()
    product_abc = analyzer.load_data(sample_data['product_data']).abc_analysis('annual_revenue', 'product_id')
    
    summary = analyzer.get_pareto_summary()
    print(f"   Key Finding: {summary['interpretation']}")
    print(f"   Pareto Ratio: {summary['pareto_ratio']} rule observed")
    
    # Show ABC distribution
    abc_dist = product_abc['abc_class'].value_counts().sort_index()
    print(f"\n   ABC Distribution:")
    for category, count in abc_dist.items():
        percent = (count / len(product_abc)) * 100
        print(f"   - Category {category}: {count} products ({percent:.1f}%)")
    
    # 2. Supplier Quality Issues Analysis
    print("\n2. Supplier Quality Issues Analysis")
    print("   Identifying suppliers causing most quality problems\n")
    
    supplier_analyzer = ParetoAnalyzer()
    supplier_abc = supplier_analyzer.load_data(sample_data['supplier_data']).abc_analysis('quality_issues', 'supplier_id')
    
    supplier_summary = supplier_analyzer.get_pareto_summary()
    print(f"   Key Finding: {supplier_summary['interpretation']}")
    
    # Show top 10 problem suppliers
    top_suppliers = supplier_abc.head(10)[['supplier_id', 'quality_issues', 'cumulative_percent']]
    print(f"\n   Top 10 Problem Suppliers:")
    for idx, row in top_suppliers.iterrows():
        print(f"   {row['supplier_id']}: {row['quality_issues']} issues ({row['cumulative_percent']:.1f}% cumulative)")
    
    print("\n3. Business Implications:")
    print("   📊 Product Focus: Concentrate marketing/inventory on Category A products")
    print("   🏭 Supplier Management: Prioritize improvement programs for problem suppliers")
    print("   💰 Resource Allocation: 80/20 rule helps prioritize where to invest time/money")
    print("   📈 Performance Monitoring: Track the vital few rather than the trivial many")
    
    return {
        'product_analysis': product_abc,
        'supplier_analysis': supplier_abc,
        'sample_data': sample_data
    }


def supply_chain_pareto_examples():
    """Provide examples of 80/20 applications in different supply chain areas."""
    
    examples = {
        "Inventory Management": [
            "80% of inventory investment in 20% of SKUs",
            "80% of stockouts from 20% of items",
            "80% of obsolete inventory from 20% of SKUs"
        ],
        
        "Supplier Management": [
            "80% of purchase spend with 20% of suppliers",
            "80% of quality issues from 20% of suppliers",
            "80% of delivery delays from 20% of suppliers"
        ],
        
        "Customer Service": [
            "80% of customer complaints about 20% of products",
            "80% of returns from 20% of product lines",
            "80% of service calls related to 20% of issues"
        ],
        
        "Logistics": [
            "80% of transportation costs from 20% of routes",
            "80% of warehouse activity in 20% of locations",
            "80% of packaging costs from 20% of product types"
        ],
        
        "Sales & Marketing": [
            "80% of revenue from 20% of customers",
            "80% of profit from 20% of product categories",
            "80% of sales volume from 20% of sales channels"
        ]
    }
    
    return examples


if __name__ == "__main__":
    # Run the demonstration
    results = demonstrate_pareto_applications()
    
    print("\n" + "="*60)
    print("📚 Additional Learning Resources:")
    print("   - Run this as a Jupyter notebook for interactive charts")
    print("   - Modify the sample data to test different scenarios")
    print("   - Apply to your own supply chain data")
    print("   - Explore optimization_problems/ for next steps")
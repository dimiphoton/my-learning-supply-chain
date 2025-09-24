"""
Retail Supply Chain Case Study

This case study demonstrates how to apply supply chain analytics to a
typical retail scenario. We'll analyze a fictional electronics retailer's
supply chain data and identify optimization opportunities.

Scenario:
TechMart is a mid-size electronics retailer with 100 product SKUs, 
10 suppliers, and seasonal demand patterns. They want to optimize
their inventory management and supplier relationships.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from supply_chain_basics.introduction import calculate_key_metrics
from pareto_analysis.pareto_basics import ParetoAnalyzer
from optimization_problems.inventory_optimization import EOQOptimizer

def load_case_study_data():
    """Load the sample data for our case study."""
    
    # Load the generated sample data
    products_df = pd.read_csv('../data/sample_products.csv')
    suppliers_df = pd.read_csv('../data/sample_suppliers.csv')
    inventory_df = pd.read_csv('../data/sample_inventory_data.csv')
    
    return products_df, suppliers_df, inventory_df

def analyze_current_situation(products_df, suppliers_df):
    """Analyze TechMart's current supply chain situation."""
    
    print("🏪 TECHMART SUPPLY CHAIN ANALYSIS")
    print("=" * 50)
    
    # Basic statistics
    total_products = len(products_df)
    total_suppliers = len(suppliers_df)
    total_annual_cost = products_df['annual_cost'].sum()
    
    print(f"\n📊 Current Situation:")
    print(f"   • Product SKUs: {total_products}")
    print(f"   • Active suppliers: {total_suppliers}")
    print(f"   • Total annual cost: ${total_annual_cost:,.0f}")
    
    # Category analysis
    category_analysis = products_df.groupby('category').agg({
        'product_id': 'count',
        'annual_cost': 'sum',
        'unit_cost': 'mean',
        'lead_time_days': 'mean'
    }).round(2)
    
    print(f"\n📦 Product Category Breakdown:")
    for category, row in category_analysis.iterrows():
        print(f"   • {category}: {int(row['product_id'])} products, ${row['annual_cost']:,.0f} annual cost")
    
    return category_analysis

def identify_optimization_opportunities(products_df, suppliers_df):
    """Use Pareto analysis to identify optimization opportunities."""
    
    print(f"\n🎯 OPTIMIZATION OPPORTUNITIES")
    print("=" * 50)
    
    # 1. Product revenue Pareto analysis
    print(f"\n1. Product Portfolio Analysis (80/20 Rule):")
    
    analyzer = ParetoAnalyzer()
    product_abc = analyzer.load_data(products_df).abc_analysis('annual_cost', 'product_id')
    summary = analyzer.get_pareto_summary()
    
    print(f"   Finding: {summary['interpretation']}")
    print(f"   Actual ratio: {summary['pareto_ratio']}")
    
    # Show Category A products (high-value)
    category_a = product_abc[product_abc['abc_class'] == 'A']
    print(f"\n   🎯 Category A Products (Focus Areas - {len(category_a)} products):")
    for _, product in category_a.head(5).iterrows():
        print(f"      • {product['product_id']}: ${product['annual_cost']:,.0f} annual cost")
    
    # 2. Supplier analysis
    print(f"\n2. Supplier Performance Analysis:")
    
    # Analyze spend concentration
    supplier_analyzer = ParetoAnalyzer()
    supplier_abc = supplier_analyzer.load_data(suppliers_df).abc_analysis('annual_spend', 'supplier_id')
    supplier_summary = supplier_analyzer.get_pareto_summary()
    
    print(f"   Finding: {supplier_summary['interpretation']}")
    
    # Show strategic suppliers
    strategic_suppliers = supplier_abc[supplier_abc['abc_class'] == 'A']
    print(f"\n   🤝 Strategic Suppliers (Category A - {len(strategic_suppliers)} suppliers):")
    for _, supplier in strategic_suppliers.iterrows():
        print(f"      • {supplier['supplier_name']}: ${supplier['annual_spend']:,.0f} spend")
        print(f"        Quality: {supplier['quality_score']}/10, Delivery: {supplier['delivery_performance']}%")
    
    return product_abc, supplier_abc

def calculate_optimization_potential(products_df, product_abc):
    """Calculate potential savings from optimization."""
    
    print(f"\n💰 OPTIMIZATION POTENTIAL")
    print("=" * 50)
    
    # EOQ analysis for Category A products
    category_a = product_abc[product_abc['abc_class'] == 'A']
    
    total_savings = 0
    print(f"\n📈 EOQ Optimization for Category A Products:")
    
    for _, product in category_a.head(3).iterrows():  # Top 3 for demo
        product_data = products_df[products_df['product_id'] == product['product_id']].iloc[0]
        
        # Assume current ordering and holding costs
        annual_demand = product_data['annual_demand']
        ordering_cost = 75  # Assumed ordering cost
        holding_cost = product_data['unit_cost'] * 0.25  # 25% holding cost
        
        eoq_optimizer = EOQOptimizer(annual_demand, ordering_cost, holding_cost)
        eoq_result = eoq_optimizer.basic_eoq()
        
        # Current cost (assume non-optimal ordering)
        current_order_qty = annual_demand / 6  # Ordering 6 times per year
        current_ordering_cost = 6 * ordering_cost
        current_holding_cost = (current_order_qty / 2) * holding_cost
        current_total_cost = current_ordering_cost + current_holding_cost
        
        savings = current_total_cost - eoq_result['total_cost']
        total_savings += savings
        
        print(f"   • {product['product_id']}:")
        print(f"     Current cost: ${current_total_cost:.0f}, Optimal: ${eoq_result['total_cost']:.0f}")
        print(f"     Potential savings: ${savings:.0f} ({savings/current_total_cost*100:.1f}%)")
    
    print(f"\n   💵 Total potential annual savings: ${total_savings:.0f}")
    
    return total_savings

def generate_recommendations(product_abc, supplier_abc, total_savings):
    """Generate actionable recommendations."""
    
    print(f"\n📋 ACTIONABLE RECOMMENDATIONS")
    print("=" * 50)
    
    category_a_products = product_abc[product_abc['abc_class'] == 'A']
    category_c_products = product_abc[product_abc['abc_class'] == 'C']
    strategic_suppliers = supplier_abc[supplier_abc['abc_class'] == 'A']
    
    print(f"\n1. 🎯 IMMEDIATE ACTIONS (90 days):")
    print(f"   • Implement EOQ for {len(category_a_products)} Category A products")
    print(f"   • Review service levels - increase to 99% for Category A items")
    print(f"   • Audit {len(strategic_suppliers)} strategic suppliers for partnership opportunities")
    print(f"   • Set up weekly monitoring for Category A products")
    
    print(f"\n2. 📈 MEDIUM-TERM IMPROVEMENTS (6 months):")
    print(f"   • Consolidate {len(category_c_products)} Category C products (consider reducing SKUs)")
    print(f"   • Implement vendor-managed inventory for top 2 strategic suppliers")
    print(f"   • Negotiate volume discounts for Category A products")
    print(f"   • Develop demand forecasting for seasonal products")
    
    print(f"\n3. 🚀 STRATEGIC INITIATIVES (12 months):")
    print(f"   • Build supply chain dashboard for real-time monitoring")
    print(f"   • Implement advanced analytics for demand sensing")
    print(f"   • Develop supplier scorecard system")
    print(f"   • Consider e-commerce fulfillment optimization")
    
    print(f"\n4. 💰 EXPECTED BENEFITS:")
    print(f"   • Inventory cost reduction: ${total_savings:.0f}/year from EOQ optimization")
    print(f"   • Service level improvement: 95% → 99% for critical products")
    print(f"   • Supplier relationships: Stronger partnerships with top suppliers")
    print(f"   • Working capital: 10-15% reduction in inventory investment")
    
def create_executive_summary():
    """Create an executive summary of findings."""
    
    print(f"\n" + "=" * 60)
    print(f"📊 EXECUTIVE SUMMARY - TECHMART SUPPLY CHAIN OPTIMIZATION")
    print(f"=" * 60)
    
    print(f"\n🔍 KEY FINDINGS:")
    print(f"   • 3% of products drive 73% of inventory costs (Classic 80/20 pattern)")
    print(f"   • 20% of suppliers account for majority of spend")
    print(f"   • Current ordering practices are sub-optimal")
    print(f"   • Opportunity for significant cost savings through better analytics")
    
    print(f"\n🎯 RECOMMENDED APPROACH:")
    print(f"   1. Focus on Category A products first (highest impact)")
    print(f"   2. Strengthen strategic supplier relationships")
    print(f"   3. Implement data-driven inventory optimization")
    print(f"   4. Build monitoring and analytics capabilities")
    
    print(f"\n📈 EXPECTED OUTCOMES:")
    print(f"   • 10-20% reduction in inventory costs")
    print(f"   • Improved service levels for critical products")
    print(f"   • Better supplier performance and relationships")
    print(f"   • Enhanced decision-making through analytics")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   • Approve Phase 1 implementation plan")
    print(f"   • Assign dedicated project team")
    print(f"   • Begin Category A product optimization")
    print(f"   • Set up performance monitoring system")

def main():
    """Run the complete TechMart case study analysis."""
    
    print("Starting TechMart Supply Chain Optimization Case Study...\n")
    
    # Load data
    products_df, suppliers_df, inventory_df = load_case_study_data()
    
    # Analyze current situation
    category_analysis = analyze_current_situation(products_df, suppliers_df)
    
    # Identify opportunities using Pareto analysis
    product_abc, supplier_abc = identify_optimization_opportunities(products_df, suppliers_df)
    
    # Calculate optimization potential
    total_savings = calculate_optimization_potential(products_df, product_abc)
    
    # Generate recommendations
    generate_recommendations(product_abc, supplier_abc, total_savings)
    
    # Executive summary
    create_executive_summary()
    
    print(f"\n" + "="*60)
    print(f"✅ Case study analysis complete!")
    print(f"📄 This analysis demonstrates practical application of:")
    print(f"   • Supply chain KPIs and benchmarking")
    print(f"   • Pareto analysis for prioritization")
    print(f"   • EOQ optimization for inventory management")
    print(f"   • Data-driven supplier management")

if __name__ == "__main__":
    main()
"""
Generate sample datasets for the supply chain learning repository.
This script creates realistic sample data for educational purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_datasets():
    """Generate all sample datasets for the learning repository."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Generate sample products dataset
    n_products = 100
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Automotive']
    
    products_data = []
    for i in range(1, n_products + 1):
        category = np.random.choice(categories)
        
        # Realistic cost distribution by category
        cost_ranges = {
            'Electronics': (20, 500),
            'Clothing': (15, 150),
            'Home & Garden': (10, 200),
            'Sports': (25, 300),
            'Books': (5, 50),
            'Automotive': (30, 800)
        }
        
        min_cost, max_cost = cost_ranges[category]
        unit_cost = round(np.random.uniform(min_cost, max_cost), 2)
        
        # Demand follows different patterns by category
        if category == 'Electronics':
            annual_demand = int(np.random.lognormal(6, 1))  # Higher volume
        elif category == 'Books':
            annual_demand = int(np.random.lognormal(4, 1.5))  # Lower volume
        else:
            annual_demand = int(np.random.lognormal(5, 1.2))  # Medium volume
            
        # Lead time varies by category
        if category == 'Electronics':
            lead_time = np.random.randint(14, 45)  # Longer lead times
        else:
            lead_time = np.random.randint(5, 21)   # Shorter lead times
        
        # Assign suppliers (10 suppliers total)
        supplier_id = f'SUP_{np.random.randint(1, 11):03d}'
        
        products_data.append({
            'product_id': f'PRD_{i:03d}',
            'category': category,
            'unit_cost': unit_cost,
            'annual_demand': annual_demand,
            'lead_time_days': lead_time,
            'supplier_id': supplier_id,
            'annual_cost': round(unit_cost * annual_demand, 2),
            'eoq_estimate': round(np.sqrt(2 * annual_demand * 50 / (unit_cost * 0.2)), 1)
        })
    
    products_df = pd.DataFrame(products_data)
    
    # 2. Generate sample suppliers dataset
    supplier_names = [
        'TechSupply Corp', 'Global Manufacturing', 'Premier Components',
        'Reliable Parts Inc', 'Quality Materials Ltd', 'Swift Logistics',
        'Industrial Solutions', 'Advanced Systems', 'Precision Works',
        'Universal Supplies'
    ]
    
    locations = [
        'China', 'Germany', 'USA', 'Japan', 'South Korea', 
        'Taiwan', 'Mexico', 'Canada', 'Singapore', 'India'
    ]
    
    suppliers_data = []
    for i in range(1, 11):
        supplier_id = f'SUP_{i:03d}'
        
        # Calculate spend based on products assigned to this supplier
        supplier_products = products_df[products_df['supplier_id'] == supplier_id]
        annual_spend = supplier_products['annual_cost'].sum()
        
        # Quality score (higher spend suppliers tend to have better quality)
        if annual_spend > 500000:
            quality_score = round(np.random.uniform(7.5, 9.5), 1)
        elif annual_spend > 200000:
            quality_score = round(np.random.uniform(6.0, 8.5), 1)
        else:
            quality_score = round(np.random.uniform(5.0, 7.5), 1)
        
        # Delivery performance
        delivery_performance = round(np.random.uniform(75, 98), 1)
        
        # Average lead time
        supplier_lead_times = supplier_products['lead_time_days'].values
        if len(supplier_lead_times) > 0:
            lead_time_avg = round(supplier_lead_times.mean(), 1)
        else:
            lead_time_avg = round(np.random.uniform(10, 20), 1)
        
        suppliers_data.append({
            'supplier_id': supplier_id,
            'supplier_name': supplier_names[i-1],
            'location': locations[i-1],
            'annual_spend': round(annual_spend, 2),
            'quality_score': quality_score,
            'delivery_performance': delivery_performance,
            'lead_time_avg': lead_time_avg,
            'num_products': len(supplier_products)
        })
    
    suppliers_df = pd.DataFrame(suppliers_data)
    
    # 3. Generate sample inventory data (6 months of daily data for top 20 products)
    top_products = products_df.nlargest(20, 'annual_demand')['product_id'].values
    
    start_date = datetime.now() - timedelta(days=180)
    dates = [start_date + timedelta(days=i) for i in range(180)]
    
    inventory_data = []
    
    for product_id in top_products:
        product_info = products_df[products_df['product_id'] == product_id].iloc[0]
        daily_demand_avg = product_info['annual_demand'] / 365
        
        # Starting inventory
        current_inventory = int(product_info['eoq_estimate'] * 1.5)
        
        for date in dates:
            # Generate daily demand (Poisson distribution around average)
            daily_demand = np.random.poisson(daily_demand_avg)
            
            # Check if stockout occurs
            stockout_flag = 1 if current_inventory < daily_demand else 0
            
            # Fulfill demand
            actual_demand = min(daily_demand, current_inventory)
            current_inventory = max(0, current_inventory - actual_demand)
            
            # Replenishment logic (simple reorder point)
            reorder_point = daily_demand_avg * product_info['lead_time_days']
            order_quantity = 0
            
            if current_inventory <= reorder_point:
                order_quantity = int(product_info['eoq_estimate'])
                # Assume inventory arrives immediately for simplicity
                current_inventory += order_quantity
            
            inventory_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'product_id': product_id,
                'inventory_level': current_inventory,
                'demand': daily_demand,
                'actual_sales': actual_demand,
                'stockout_flag': stockout_flag,
                'order_quantity': order_quantity,
                'reorder_point': round(reorder_point, 1)
            })
    
    inventory_df = pd.DataFrame(inventory_data)
    
    return products_df, suppliers_df, inventory_df

if __name__ == "__main__":
    print("🏭 Generating sample supply chain datasets...")
    
    products_df, suppliers_df, inventory_df = generate_sample_datasets()
    
    # Save datasets
    products_df.to_csv('sample_products.csv', index=False)
    suppliers_df.to_csv('sample_suppliers.csv', index=False) 
    inventory_df.to_csv('sample_inventory_data.csv', index=False)
    
    print("✅ Sample datasets generated successfully!")
    print(f"   📦 Products: {len(products_df)} records")
    print(f"   🏭 Suppliers: {len(suppliers_df)} records") 
    print(f"   📊 Inventory: {len(inventory_df)} records")
    
    # Show summary statistics
    print(f"\n📈 Dataset Summary:")
    print(f"   Total annual spend: ${suppliers_df['annual_spend'].sum():,.0f}")
    print(f"   Average quality score: {suppliers_df['quality_score'].mean():.1f}/10")
    print(f"   Product categories: {products_df['category'].nunique()}")
    print(f"   Inventory data period: 6 months for top 20 products")
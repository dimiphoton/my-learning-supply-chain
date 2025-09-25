# Supply Chain Sample Datasets

This directory contains sample datasets for learning supply chain analytics and data science techniques.

## Available Datasets

### 1. sample_products.csv
Sample product data with the following columns:
- `product_id`: Unique product identifier
- `category`: Product category (Electronics, Clothing, etc.)
- `unit_cost`: Cost per unit in USD
- `annual_demand`: Expected annual demand in units
- `lead_time_days`: Lead time from order to receipt
- `supplier_id`: Associated supplier identifier
- `annual_cost`: Total annual cost (unit_cost × annual_demand)
- `eoq_estimate`: Estimated Economic Order Quantity

### 2. sample_suppliers.csv
Sample supplier data with columns:
- `supplier_id`: Unique supplier identifier
- `supplier_name`: Company name
- `location`: Geographic location
- `annual_spend`: Total annual spending with this supplier
- `quality_score`: Quality rating (1-10 scale)
- `delivery_performance`: On-time delivery percentage
- `lead_time_avg`: Average lead time in days

### 3. sample_inventory_data.csv
Historical inventory data for analysis:
- `date`: Date of record
- `product_id`: Product identifier
- `inventory_level`: Stock on hand
- `demand`: Daily demand
- `stockout_flag`: Whether stockout occurred (0/1)
- `order_quantity`: Replenishment order size

## Usage Notes

- These are synthetic datasets created for educational purposes
- Data follows realistic supply chain patterns and distributions
- Perfect for learning without confidentiality concerns
- Use these as templates for structuring your own data

## Loading Data in Python

```python
import pandas as pd

# Load product data
products = pd.read_csv('data/sample_products.csv')

# Load supplier data  
suppliers = pd.read_csv('data/sample_suppliers.csv')

# Load inventory data
inventory = pd.read_csv('data/sample_inventory_data.csv')
```

## Data Generation

The datasets are generated using realistic parameters and distributions commonly found in supply chain operations. Random seeds are used to ensure reproducibility for learning exercises.
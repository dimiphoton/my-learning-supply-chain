"""
Supply Chain Basics - Core Concepts and Definitions

This module introduces fundamental supply chain concepts for data scientists.
Understanding these concepts is crucial for applying data science techniques
to supply chain problems.
"""

class SupplyChainComponent:
    """Represents a component in the supply chain."""
    
    def __init__(self, name, type_of_component, description):
        self.name = name
        self.type = type_of_component
        self.description = description
        self.upstream_partners = []
        self.downstream_partners = []
    
    def add_upstream_partner(self, partner):
        """Add an upstream supply chain partner."""
        self.upstream_partners.append(partner)
    
    def add_downstream_partner(self, partner):
        """Add a downstream supply chain partner."""
        self.downstream_partners.append(partner)
    
    def get_info(self):
        """Return information about this supply chain component."""
        return {
            'name': self.name,
            'type': self.type,
            'description': self.description,
            'upstream_count': len(self.upstream_partners),
            'downstream_count': len(self.downstream_partners)
        }


def create_basic_supply_chain():
    """
    Create a basic supply chain structure to demonstrate concepts.
    
    Returns:
        dict: Dictionary containing supply chain components
    """
    
    # Create supply chain components
    supplier = SupplyChainComponent(
        "Raw Material Supplier",
        "Supplier",
        "Provides raw materials needed for production"
    )
    
    manufacturer = SupplyChainComponent(
        "Manufacturing Plant",
        "Manufacturer",
        "Converts raw materials into finished goods"
    )
    
    distributor = SupplyChainComponent(
        "Distribution Center",
        "Distributor",
        "Stores and distributes finished goods to retailers"
    )
    
    retailer = SupplyChainComponent(
        "Retail Store",
        "Retailer",
        "Sells products directly to end customers"
    )
    
    customer = SupplyChainComponent(
        "End Customer",
        "Customer",
        "Final consumer of the product"
    )
    
    # Establish relationships
    supplier.add_downstream_partner(manufacturer)
    manufacturer.add_upstream_partner(supplier)
    manufacturer.add_downstream_partner(distributor)
    distributor.add_upstream_partner(manufacturer)
    distributor.add_downstream_partner(retailer)
    retailer.add_upstream_partner(distributor)
    retailer.add_downstream_partner(customer)
    customer.add_upstream_partner(retailer)
    
    return {
        'supplier': supplier,
        'manufacturer': manufacturer,
        'distributor': distributor,
        'retailer': retailer,
        'customer': customer
    }


def get_supply_chain_definitions():
    """
    Return key supply chain definitions and concepts.
    
    Returns:
        dict: Dictionary of supply chain terms and definitions
    """
    
    definitions = {
        "Supply Chain": {
            "definition": "A network of organizations, people, activities, information, and resources involved in moving a product or service from supplier to customer.",
            "importance": "Understanding the entire flow helps identify bottlenecks and optimization opportunities."
        },
        
        "Lead Time": {
            "definition": "The time it takes from placing an order until the product is received.",
            "importance": "Critical for inventory planning and customer satisfaction."
        },
        
        "Bullwhip Effect": {
            "definition": "The amplification of demand variability as you move upstream in the supply chain.",
            "importance": "Understanding this helps in demand planning and inventory management."
        },
        
        "Safety Stock": {
            "definition": "Extra inventory held to guard against uncertainty in supply and demand.",
            "importance": "Balances service level with inventory costs."
        },
        
        "Cycle Stock": {
            "definition": "Inventory that is ordered to meet expected demand during the replenishment cycle.",
            "importance": "Core component of inventory optimization models."
        },
        
        "SKU (Stock Keeping Unit)": {
            "definition": "A unique identifier for each distinct product and service that can be purchased.",
            "importance": "Fundamental unit for inventory management and analysis."
        },
        
        "Vendor Managed Inventory (VMI)": {
            "definition": "A supply chain practice where the supplier manages inventory levels at the customer's location.",
            "importance": "Can reduce costs and improve service levels through better coordination."
        },
        
        "Cross-docking": {
            "definition": "A logistics practice where incoming goods are directly transferred to outgoing transportation with minimal or no storage.",
            "importance": "Reduces inventory costs and improves speed to market."
        }
    }
    
    return definitions


def calculate_key_metrics(revenue, cogs, inventory_value, lead_time_days):
    """
    Calculate key supply chain performance metrics.
    
    Args:
        revenue (float): Annual revenue
        cogs (float): Cost of goods sold
        inventory_value (float): Current inventory value
        lead_time_days (int): Average lead time in days
    
    Returns:
        dict: Dictionary of calculated metrics
    """
    
    # Inventory turnover ratio
    inventory_turnover = cogs / inventory_value if inventory_value > 0 else 0
    
    # Days of supply
    days_of_supply = (inventory_value / cogs) * 365 if cogs > 0 else 0
    
    # Gross margin
    gross_margin = ((revenue - cogs) / revenue) * 100 if revenue > 0 else 0
    
    # Inventory as % of revenue
    inventory_percentage = (inventory_value / revenue) * 100 if revenue > 0 else 0
    
    return {
        "inventory_turnover": round(inventory_turnover, 2),
        "days_of_supply": round(days_of_supply, 1),
        "gross_margin_percent": round(gross_margin, 1),
        "inventory_as_percent_revenue": round(inventory_percentage, 1),
        "lead_time_days": lead_time_days
    }


if __name__ == "__main__":
    # Demonstrate the supply chain concepts
    print("=== Supply Chain Learning Repository ===\n")
    
    # Create and display a basic supply chain
    print("1. Basic Supply Chain Structure:")
    supply_chain = create_basic_supply_chain()
    
    for component_name, component in supply_chain.items():
        info = component.get_info()
        print(f"   {info['name']}: {info['description']}")
    
    print("\n2. Key Supply Chain Definitions:")
    definitions = get_supply_chain_definitions()
    
    for term, details in list(definitions.items())[:3]:  # Show first 3
        print(f"   {term}: {details['definition']}")
    
    print(f"\n   ... and {len(definitions)-3} more definitions available")
    
    print("\n3. Example Metrics Calculation:")
    example_metrics = calculate_key_metrics(
        revenue=1000000,      # $1M annual revenue
        cogs=600000,          # $600K cost of goods sold
        inventory_value=150000, # $150K inventory
        lead_time_days=14      # 2 weeks lead time
    )
    
    for metric, value in example_metrics.items():
        print(f"   {metric.replace('_', ' ').title()}: {value}")
    
    print("\n🎯 Next Steps:")
    print("   - Explore pareto_analysis/ for 80/20 principle applications")
    print("   - Check optimization_problems/ for practical optimization examples")
    print("   - Run Jupyter notebooks for hands-on practice")
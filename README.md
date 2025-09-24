# Supply Chain Data Science Learning Repository

A comprehensive learning repository for data scientists focusing on supply chain analytics, optimization, and the Pareto (80/20) principle.

## 📚 What You'll Learn

### 1. **Supply Chain Fundamentals**
- **What is Supply Chain?** - Understanding the flow of goods, information, and finances from raw materials to end customers
- **Key Components**: Suppliers, Manufacturers, Distributors, Retailers, Customers
- **Core Processes**: Procurement, Production, Distribution, Inventory Management
- **Performance Metrics**: Cost, Quality, Speed, Flexibility, Sustainability

### 2. **The 80/20 Rule (Pareto Principle) in Supply Chain**
- **What is the 80/20 Rule?** - 80% of effects come from 20% of causes
- **Applications in Supply Chain**:
  - 80% of revenue from 20% of products
  - 80% of inventory costs from 20% of SKUs
  - 80% of supply issues from 20% of suppliers
  - 80% of customer complaints from 20% of processes

### 3. **Supply Chain Optimization Problems**
- **Inventory Optimization**: Minimizing holding costs while maintaining service levels
- **Transportation Optimization**: Finding optimal routes and modes
- **Network Design**: Optimal facility locations and capacity planning
- **Demand Forecasting**: Using ML to predict future demand patterns
- **Supplier Selection**: Multi-criteria decision making

## 🗂️ Repository Structure

```
├── src/
│   ├── supply_chain_basics/     # Core concepts and theory
│   ├── pareto_analysis/         # 80/20 principle implementations
│   └── optimization_problems/   # Optimization algorithms and examples
├── notebooks/                   # Jupyter notebooks for interactive learning
├── data/                       # Sample datasets
├── examples/                   # Practical examples and case studies
└── requirements.txt            # Python dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Basic knowledge of Python and data analysis
- Understanding of business concepts is helpful but not required

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dimiphoton/my-learning-supply-chain.git
   cd my-learning-supply-chain
   ```

2. **Set up virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter notebooks**:
   ```bash
   jupyter notebook
   ```

## 📖 Learning Path

### Beginner Level
1. **Start with Supply Chain Basics** (`src/supply_chain_basics/`)
   - `introduction.py` - Core concepts and terminology
   - `supply_chain_components.py` - Understanding each component

2. **Explore the 80/20 Principle** (`src/pareto_analysis/`)
   - `pareto_basics.py` - Understanding the Pareto principle
   - `supply_chain_applications.py` - Real-world applications

3. **Practice with Notebooks** (`notebooks/`)
   - `01_supply_chain_introduction.ipynb`
   - `02_pareto_analysis_tutorial.ipynb`

### Intermediate Level
4. **Basic Optimization** (`src/optimization_problems/`)
   - `inventory_optimization.py` - EOQ models and variants
   - `transportation_problem.py` - Classical transportation optimization

5. **Data Analysis Projects** (`notebooks/`)
   - `03_inventory_analysis.ipynb`
   - `04_supplier_performance.ipynb`

### Advanced Level
6. **Machine Learning Applications**
   - `demand_forecasting.py` - Time series and ML forecasting
   - `predictive_maintenance.py` - ML for equipment reliability

7. **Complex Optimization**
   - `network_optimization.py` - Multi-echelon network design
   - `multi_objective_optimization.py` - Balancing multiple criteria

## 🛠️ Tools and Libraries Used

- **Data Analysis**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn
- **Optimization**: PuLP, CVXPY, OR-Tools
- **Interactive Learning**: Jupyter notebooks

## 📊 Key Concepts Covered

### Supply Chain Management
- **Bullwhip Effect**: How small demand changes amplify upstream
- **Supply Chain Visibility**: Importance of real-time information
- **Risk Management**: Identifying and mitigating supply chain risks
- **Sustainability**: Green supply chain practices

### Data Science Applications
- **ABC Analysis**: Categorizing items by importance
- **Demand Sensing**: Using multiple data sources for better forecasting
- **Optimization Models**: Linear programming, integer programming
- **Performance Analytics**: KPIs and dashboards

## 🎯 Learning Outcomes

By completing this repository, you will be able to:

1. **Understand** supply chain fundamentals and key terminology
2. **Apply** the 80/20 principle to identify critical areas for improvement
3. **Implement** optimization algorithms for common supply chain problems
4. **Analyze** supply chain data using Python and visualization tools
5. **Build** predictive models for demand forecasting and risk assessment
6. **Design** data-driven solutions for supply chain challenges

## 🤝 Contributing

This is a learning repository. Feel free to:
- Add new examples or case studies
- Improve existing code or documentation
- Share interesting datasets
- Report bugs or suggest improvements

## 📚 Additional Resources

- [Supply Chain Management by Sunil Chopra](https://www.amazon.com/Supply-Chain-Management-Strategy-Planning/dp/0133800202)
- [Introduction to Operations Research by Hillier & Lieberman](https://www.amazon.com/Introduction-Operations-Research-Frederick-Hillier/dp/0073523453)
- [Python for Data Analysis by Wes McKinney](https://wesmckinney.com/book/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy Learning! 🚀**

*This repository is designed to bridge the gap between supply chain theory and practical data science implementation.*

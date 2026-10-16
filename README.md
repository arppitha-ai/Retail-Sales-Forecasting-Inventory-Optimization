# Retail Sales Forecasting & Inventory Optimization System

An end-to-end data science project combining retail sales forecasting, inventory optimization, and inventory simulation to support data-driven inventory planning.

The project uses simulated retail data to demonstrate how historical sales patterns can be analyzed to forecast future demand, identify products that may need replenishment, and evaluate inventory performance.

> **Important:** All retail data and business results are simulated for educational and portfolio demonstration purposes. They do not represent a real retailer.

## Project Overview

Retail businesses must maintain enough inventory to meet customer demand without holding excessive stock. Poor inventory planning can contribute to stockouts, lost sales, and unnecessary inventory costs.

This project demonstrates a data-driven approach through:

- Retail sales data generation and preprocessing
- Exploratory Data Analysis (EDA)
- Machine learning-based sales forecasting
- Forecast model evaluation
- Safety stock and reorder point calculations
- Inventory replenishment recommendations
- Virtual inventory simulation
- Interactive Streamlit dashboard
- Automated data validation tests

## Project Objectives

1. Analyze historical sales patterns across multiple stores and products.
2. Forecast short-term product demand.
3. Compare forecasting models using MAE and RMSE.
4. Estimate inventory requirements using demand forecasts and lead times.
5. Recommend replenishment quantities.
6. Simulate inventory operations and estimate lost sales.
7. Present results through an interactive dashboard.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical calculations |
| Scikit-learn | Machine learning |
| Matplotlib | Data visualization |
| Seaborn | Exploratory data visualization |
| Plotly | Interactive charts |
| Streamlit | Interactive dashboard |
| Pytest | Automated testing |
| CSV | Data storage and exchange |

## Project Structure

```text
Retail-Sales-Forecasting-Inventory-Optimization/
├── app/
│   └── dashboard.py
├── data/
│   ├── processed/
│   │   └── cleaned_sales.csv
│   └── simulated/
│       ├── sales_history.csv
│       └── inventory.csv
├── outputs/
│   ├── figures/
│   ├── forecasts/
│   ├── inventory/
│   └── metrics/
├── src/
│   ├── config.py
│   ├── data_simulator.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── forecasting.py
│   ├── inventory_optimizer.py
│   └── inventory_simulator.py
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
### Dashboard Preview

#### 1. Business Overview

![Business Overview](images/dashboard/01_dashboard_overview.png)

#### 2. Sales Analysis

![Sales Analysis](images/dashboard/02_sales_analysis.png)

#### 3. Sales Forecasting

![Sales Forecasting](images/dashboard/03_sales_forecasting.png)

#### 4. Inventory Optimization

![Inventory Optimization](images/dashboard/04_inventory_optimization.png)

#### 5. Inventory Simulation

![Inventory Simulation](images/dashboard/05_inventory_simulation.png)
"""Generate reproducible synthetic retail sales and inventory data."""

import numpy as np
import pandas as pd

from src.config import (
    HISTORY_DAYS,
    NUM_PRODUCTS,
    NUM_STORES,
    RANDOM_SEED,
    SIMULATED_DATA_DIR,
)


PRODUCTS = [
    {"product_id": "P001", "product_name": "Rice 5kg", "category": "Grocery", "unit_price": 420.0},
    {"product_id": "P002", "product_name": "Cooking Oil 1L", "category": "Grocery", "unit_price": 155.0},
    {"product_id": "P003", "product_name": "Milk 1L", "category": "Dairy", "unit_price": 65.0},
    {"product_id": "P004", "product_name": "Bread", "category": "Bakery", "unit_price": 45.0},
    {"product_id": "P005", "product_name": "Soap", "category": "Personal Care", "unit_price": 38.0},
    {"product_id": "P006", "product_name": "Shampoo", "category": "Personal Care", "unit_price": 120.0},
    {"product_id": "P007", "product_name": "Biscuits", "category": "Snacks", "unit_price": 30.0},
    {"product_id": "P008", "product_name": "Coffee 200g", "category": "Beverages", "unit_price": 180.0},
]

BASE_DEMAND = {
    "P001": 18,
    "P002": 15,
    "P003": 24,
    "P004": 20,
    "P005": 12,
    "P006": 8,
    "P007": 28,
    "P008": 10,
}


def generate_sales_data() -> pd.DataFrame:
    """Generate daily sales for every store and product."""
    rng = np.random.default_rng(RANDOM_SEED)

    end_date = pd.Timestamp.today().normalize()
    dates = pd.date_range(
        end=end_date,
        periods=HISTORY_DAYS,
        freq="D",
    )

    rows = []

    for store_number in range(1, NUM_STORES + 1):
        store_id = f"S{store_number:02d}"

        for product in PRODUCTS[:NUM_PRODUCTS]:
            product_id = product["product_id"]
            base_demand = BASE_DEMAND[product_id]

            # Store-level variation
            store_factor = 0.80 + (store_number * 0.12)

            for date in dates:
                # Weekly demand pattern
                weekday_factor = 1.20 if date.dayofweek in (4, 5) else 1.0

                # Gradual trend and mild seasonal variation
                day_number = (date - dates[0]).days
                trend_factor = 1.0 + (0.0005 * day_number)
                seasonal_factor = (
                    1.0 + 0.10 * np.sin(2 * np.pi * day_number / 30)
                )

                expected_demand = (
                    base_demand
                    * store_factor
                    * weekday_factor
                    * trend_factor
                    * seasonal_factor
                )

                units_sold = int(rng.poisson(max(expected_demand, 0.1)))

                # Occasional zero-demand days
                if rng.random() < 0.04:
                    units_sold = 0

                unit_price = product["unit_price"]
                revenue = round(units_sold * unit_price, 2)

                rows.append(
                    {
                        "date": date,
                        "store_id": store_id,
                        "product_id": product_id,
                        "product_name": product["product_name"],
                        "category": product["category"],
                        "units_sold": units_sold,
                        "unit_price": unit_price,
                        "revenue": revenue,
                    }
                )

    sales = pd.DataFrame(rows)
    return sales.sort_values(
        ["date", "store_id", "product_id"]
    ).reset_index(drop=True)


def generate_inventory_data() -> pd.DataFrame:
    """Generate a starting inventory snapshot for each store and product."""
    rng = np.random.default_rng(RANDOM_SEED + 1)
    rows = []

    for store_number in range(1, NUM_STORES + 1):
        store_id = f"S{store_number:02d}"

        for product in PRODUCTS[:NUM_PRODUCTS]:
            product_id = product["product_id"]

            rows.append(
                {
                    "store_id": store_id,
                    "product_id": product_id,
                    "on_hand": int(rng.integers(10, 100)),
                    "lead_time_days": int(rng.integers(2, 10)),
                    "unit_cost": round(product["unit_price"] * 0.70, 2),
                }
            )

    return pd.DataFrame(rows)


def save_synthetic_data() -> None:
    """Generate and save sales and inventory CSV files."""
    SIMULATED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    sales = generate_sales_data()
    inventory = generate_inventory_data()

    sales_path = SIMULATED_DATA_DIR / "sales_history.csv"
    inventory_path = SIMULATED_DATA_DIR / "inventory.csv"

    sales.to_csv(sales_path, index=False)
    inventory.to_csv(inventory_path, index=False)

    print("=" * 55)
    print("SYNTHETIC RETAIL DATA GENERATION COMPLETE")
    print("=" * 55)
    print(f"Sales records: {len(sales):,}")
    print(f"Stores: {sales['store_id'].nunique()}")
    print(f"Products: {sales['product_id'].nunique()}")
    print(f"Date range: {sales['date'].min().date()} to {sales['date'].max().date()}")
    print(f"Total units sold: {sales['units_sold'].sum():,}")
    print(f"Total simulated revenue: {sales['revenue'].sum():,.2f}")
    print(f"\nSales file: {sales_path}")
    print(f"Inventory file: {inventory_path}")
    print("\nNote: These records are synthetic, not real retailer data.")


if __name__ == "__main__":
    save_synthetic_data()
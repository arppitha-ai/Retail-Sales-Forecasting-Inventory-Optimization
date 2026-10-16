"""
Retail Sales Forecasting & Inventory Optimization System

Calculates inventory recommendations using historical sales,
forecast demand, lead time, safety stock, and reorder policies.

IMPORTANT:
All data used in this project is synthetic and simulated.
Recommendations are for educational and portfolio purposes.
"""

import numpy as np
import pandas as pd

from src.config import (
    FORECAST_DIR,
    INVENTORY_DIR,
    PROCESSED_DATA_DIR,
    SIMULATED_DATA_DIR,
)


def recommend_inventory(
    sales: pd.DataFrame,
    forecasts: pd.DataFrame,
    inventory: pd.DataFrame,
    review_period_days: int = 7,
    service_level_z: float = 1.65,
) -> pd.DataFrame:
    """
    Generate inventory recommendations for every store-product pair.

    Calculations:
    1. Historical average daily demand
    2. Historical demand standard deviation
    3. Forecast demand during lead time
    4. Safety stock
    5. Reorder point
    6. Target stock level
    7. Suggested order quantity
    8. Estimated order cost

    Days beyond the available forecast horizon use historical
    average daily demand as a planning assumption.
    """

    # ---------------------------------------------------------
    # 1. Validate the input parameters
    # ---------------------------------------------------------

    if review_period_days < 1:
        raise ValueError(
            "Review period must be at least one day."
        )

    if service_level_z < 0:
        raise ValueError(
            "Service-level factor cannot be negative."
        )

    # ---------------------------------------------------------
    # 2. Validate required columns
    # ---------------------------------------------------------

    required_sales_columns = {
        "store_id",
        "product_id",
        "date",
        "units_sold",
    }

    required_forecast_columns = {
        "store_id",
        "product_id",
        "date",
        "predicted_units",
    }

    required_inventory_columns = {
        "store_id",
        "product_id",
        "on_hand",
        "lead_time_days",
        "unit_cost",
    }

    if not required_sales_columns.issubset(sales.columns):
        missing = required_sales_columns - set(sales.columns)
        raise ValueError(
            f"Sales data is missing columns: {sorted(missing)}"
        )

    if not required_forecast_columns.issubset(forecasts.columns):
        missing = (
            required_forecast_columns - set(forecasts.columns)
        )
        raise ValueError(
            f"Forecast data is missing columns: {sorted(missing)}"
        )

    if not required_inventory_columns.issubset(inventory.columns):
        missing = (
            required_inventory_columns - set(inventory.columns)
        )
        raise ValueError(
            f"Inventory data is missing columns: {sorted(missing)}"
        )

    # Work on copies so the original DataFrames are unchanged.
    sales = sales.copy()
    forecasts = forecasts.copy()
    inventory = inventory.copy()

    # ---------------------------------------------------------
    # 3. Convert data types and validate values
    # ---------------------------------------------------------

    sales["date"] = pd.to_datetime(
        sales["date"],
        errors="raise",
    )

    forecasts["date"] = pd.to_datetime(
        forecasts["date"],
        errors="raise",
    )

    numeric_columns = {
        "sales": ["units_sold"],
        "forecasts": ["predicted_units"],
        "inventory": [
            "on_hand",
            "lead_time_days",
            "unit_cost",
        ],
    }

    for column in numeric_columns["sales"]:
        sales[column] = pd.to_numeric(
            sales[column],
            errors="raise",
        )

    for column in numeric_columns["forecasts"]:
        forecasts[column] = pd.to_numeric(
            forecasts[column],
            errors="raise",
        )

    for column in numeric_columns["inventory"]:
        inventory[column] = pd.to_numeric(
            inventory[column],
            errors="raise",
        )

    if sales[list(required_sales_columns)].isna().any().any():
        raise ValueError(
            "Sales data contains missing required values."
        )

    if forecasts[list(required_forecast_columns)].isna().any().any():
        raise ValueError(
            "Forecast data contains missing required values."
        )

    if inventory[list(required_inventory_columns)].isna().any().any():
        raise ValueError(
            "Inventory data contains missing required values."
        )

    if (sales["units_sold"] < 0).any():
        raise ValueError(
            "Historical sales quantities cannot be negative."
        )

    if (forecasts["predicted_units"] < 0).any():
        raise ValueError(
            "Forecast demand cannot be negative."
        )

    if (inventory["on_hand"] < 0).any():
        raise ValueError(
            "On-hand inventory cannot be negative."
        )

    if (inventory["lead_time_days"] < 1).any():
        raise ValueError(
            "Lead time must be at least one day."
        )

    if (inventory["unit_cost"] < 0).any():
        raise ValueError(
            "Unit cost cannot be negative."
        )

    if not np.isfinite(
        sales["units_sold"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Sales quantities must be finite numbers."
        )

    if not np.isfinite(
        forecasts["predicted_units"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Forecast demand must be finite numbers."
        )

    if not np.isfinite(
        inventory[
            ["on_hand", "lead_time_days", "unit_cost"]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Inventory values must be finite numbers."
        )

    if not np.all(
        inventory["lead_time_days"]
        == np.floor(inventory["lead_time_days"])
    ):
        raise ValueError(
            "Lead time must contain whole numbers of days."
        )

    if inventory.duplicated(
        ["store_id", "product_id"]
    ).any():
        raise ValueError(
            "Inventory must contain only one row per "
            "store-product combination."
        )

    if forecasts.duplicated(
        ["store_id", "product_id", "date"]
    ).any():
        raise ValueError(
            "Forecast data contains duplicate dates "
            "for a store-product combination."
        )

    # ---------------------------------------------------------
    # 4. Calculate historical demand statistics
    # ---------------------------------------------------------

    daily_stats = (
        sales.groupby(
            ["store_id", "product_id"]
        )["units_sold"]
        .agg(
            historical_mean="mean",
            historical_std="std",
        )
        .reset_index()
    )

    # A standard deviation can be missing when only one
    # historical observation exists.
    daily_stats["historical_mean"] = (
        daily_stats["historical_mean"].fillna(0)
    )

    daily_stats["historical_std"] = (
        daily_stats["historical_std"].fillna(0)
    )

    stats_lookup = {
        (row.store_id, row.product_id): (
            float(row.historical_mean),
            float(row.historical_std),
        )
        for row in daily_stats.itertuples(index=False)
    }

    # ---------------------------------------------------------
    # 5. Prepare forecasts in chronological order
    # ---------------------------------------------------------

    forecasts = forecasts.sort_values(
        ["store_id", "product_id", "date"]
    )

    forecast_groups = {
        key: group["predicted_units"].astype(float).tolist()
        for key, group in forecasts.groupby(
            ["store_id", "product_id"]
        )
    }

    # Verify that inventory pairs have both historical
    # sales statistics and future forecasts.
    historical_pairs = set(stats_lookup)
    forecast_pairs = set(forecast_groups)

    results = []

    # ---------------------------------------------------------
    # 6. Calculate inventory recommendations
    # ---------------------------------------------------------

    for row in inventory.itertuples(index=False):

        store_id = row.store_id
        product_id = row.product_id

        key = (store_id, product_id)

        if key not in historical_pairs:
            raise ValueError(
                "No historical sales found for "
                f"store {store_id}, product {product_id}."
            )

        if key not in forecast_pairs:
            raise ValueError(
                "No forecasts found for "
                f"store {store_id}, product {product_id}."
            )

        historical_mean, historical_std = stats_lookup[key]

        daily_forecasts = forecast_groups[key]

        if not daily_forecasts:
            raise ValueError(
                f"Forecast list is empty for {key}."
            )

        lead_time = int(row.lead_time_days)

        # The target stock covers both the supplier lead time
        # and the next inventory review period.
        planning_days = lead_time + review_period_days

        # Use historical average demand for planning days
        # beyond the available forecast horizon.
        extended_forecasts = daily_forecasts.copy()

        if len(extended_forecasts) < planning_days:

            missing_days = (
                planning_days - len(extended_forecasts)
            )

            extended_forecasts.extend(
                [historical_mean] * missing_days
            )

        # Expected demand during supplier lead time.
        lead_time_forecasts = extended_forecasts[:lead_time]

        forecast_lead_time_demand = float(
            sum(lead_time_forecasts)
        )

        # Expected demand during lead time plus review period.
        planning_forecasts = extended_forecasts[:planning_days]

        forecast_planning_demand = float(
            sum(planning_forecasts)
        )

        # -----------------------------------------------------
        # Safety stock
        # -----------------------------------------------------
        # This is a simplified planning formula:
        # z × historical daily standard deviation × sqrt(lead time)
        #
        # The z-value is a planning assumption, not a guarantee
        # that the desired service level will be achieved.

        safety_stock = int(
            np.ceil(
                service_level_z
                * historical_std
                * np.sqrt(lead_time)
            )
        )

        # -----------------------------------------------------
        # Reorder point
        # -----------------------------------------------------
        # Reorder when on-hand inventory is at or below
        # the estimated demand during lead time plus safety stock.

        reorder_point = int(
            np.ceil(
                forecast_lead_time_demand + safety_stock
            )
        )

        # -----------------------------------------------------
        # Target stock
        # -----------------------------------------------------

        target_stock = int(
            np.ceil(
                forecast_planning_demand + safety_stock
            )
        )

        on_hand = int(row.on_hand)
        unit_cost = float(row.unit_cost)

        # Do not suggest a negative order quantity.
        suggested_order_qty = max(
            0,
            target_stock - on_hand,
        )

        estimated_order_cost = (
            suggested_order_qty * unit_cost
        )

        reorder_required = (
            on_hand <= reorder_point
        )

        # -----------------------------------------------------
        # 7. Store the results
        # -----------------------------------------------------

        results.append(
            {
                "store_id": store_id,
                "product_id": product_id,
                "on_hand": on_hand,
                "lead_time_days": lead_time,
                "review_period_days": review_period_days,
                "historical_mean_daily_demand": round(
                    historical_mean,
                    2,
                ),
                "historical_daily_demand_std": round(
                    historical_std,
                    2,
                ),
                "forecast_lead_time_demand": round(
                    forecast_lead_time_demand,
                    2,
                ),
                "forecast_planning_demand": round(
                    forecast_planning_demand,
                    2,
                ),
                "safety_stock": safety_stock,
                "reorder_point": reorder_point,
                "target_stock": target_stock,
                "suggested_order_qty": suggested_order_qty,
                "unit_cost": round(unit_cost, 2),
                "estimated_order_cost": round(
                    estimated_order_cost,
                    2,
                ),
                "reorder_required": reorder_required,
            }
        )

    # ---------------------------------------------------------
    # 8. Create and sort the recommendation table
    # ---------------------------------------------------------

    recommendations = pd.DataFrame(results)

    recommendations = recommendations.sort_values(
        ["reorder_required", "suggested_order_qty"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return recommendations


def run_inventory_optimization() -> pd.DataFrame:
    """
    Load the project datasets, calculate recommendations,
    and save the results as a CSV file.
    """

    sales_path = (
        PROCESSED_DATA_DIR / "cleaned_sales.csv"
    )

    forecast_path = (
        FORECAST_DIR / "sales_forecast.csv"
    )

    inventory_path = (
        SIMULATED_DATA_DIR / "inventory.csv"
    )

    # ---------------------------------------------------------
    # 1. Check that all required files exist
    # ---------------------------------------------------------

    required_files = [
        sales_path,
        forecast_path,
        inventory_path,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Please run the data generation, "
                "preprocessing, and forecasting steps first."
            )

    # ---------------------------------------------------------
    # 2. Load the datasets
    # ---------------------------------------------------------

    sales = pd.read_csv(
        sales_path,
        parse_dates=["date"],
    )

    forecasts = pd.read_csv(
        forecast_path,
        parse_dates=["date"],
    )

    inventory = pd.read_csv(
        inventory_path,
    )

    # ---------------------------------------------------------
    # 3. Generate recommendations
    # ---------------------------------------------------------

    recommendations = recommend_inventory(
        sales=sales,
        forecasts=forecasts,
        inventory=inventory,
        review_period_days=7,
        service_level_z=1.65,
    )

    # ---------------------------------------------------------
    # 4. Save the recommendations
    # ---------------------------------------------------------

    INVENTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        INVENTORY_DIR / "inventory_recommendations.csv"
    )

    recommendations.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 5. Display a summary
    # ---------------------------------------------------------

    print("=" * 70)
    print("RETAIL INVENTORY OPTIMIZATION RESULTS")
    print("=" * 70)

    print(
        f"Store-product combinations: "
        f"{len(recommendations)}"
    )

    print(
        "Products requiring reorder: "
        f"{int(recommendations['reorder_required'].sum())}"
    )

    print(
        "Total suggested units to order: "
        f"{int(recommendations['suggested_order_qty'].sum())}"
    )

    print(
        "Estimated total order cost: "
        f"{recommendations['estimated_order_cost'].sum():,.2f}"
    )

    print("\nINVENTORY RECOMMENDATIONS")
    print("-" * 70)

    display_columns = [
        "store_id",
        "product_id",
        "on_hand",
        "safety_stock",
        "reorder_point",
        "target_stock",
        "suggested_order_qty",
        "reorder_required",
    ]

    print(
        recommendations[display_columns].to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("IMPORTANT")
    print("=" * 70)

    print(
        "These recommendations use synthetic sales and "
        "inventory data."
    )

    print(
        "They are educational planning estimates, "
        "not real purchasing instructions."
    )

    print(f"\nSaved recommendations to:\n{output_path}")

    return recommendations


if __name__ == "__main__":
    run_inventory_optimization()
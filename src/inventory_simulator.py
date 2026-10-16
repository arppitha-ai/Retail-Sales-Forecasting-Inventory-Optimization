"""
Inventory Simulation Module

Simulates daily inventory operations using synthetic retail data.

Features:
- Daily customer demand
- Inventory stock reduction
- Lost sales caused by stockouts
- Reorder-point monitoring
- Supplier lead times
- Replenishment orders and deliveries
- Inventory performance summary

IMPORTANT:
This is an educational simulation using synthetic data.
It does not represent a real retail business.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from src.config import (
    INVENTORY_DIR,
    PROCESSED_DATA_DIR,
    SIMULATED_DATA_DIR,
    RANDOM_SEED,
)


def simulate_inventory(
    sales: pd.DataFrame,
    inventory: pd.DataFrame,
    review_period_days: int = 7,
    service_level_z: float = 1.65,
    simulation_days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate inventory operations over a selected number of days.

    Historical sales patterns are used to simulate future demand.

    Returns:
        daily_results: Daily inventory activity for every product/store.
        summary: Aggregated inventory performance for every product/store.
    """

    # ---------------------------------------------------------
    # 1. Validate parameters
    # ---------------------------------------------------------

    if review_period_days < 1:
        raise ValueError("Review period must be at least one day.")

    if service_level_z < 0:
        raise ValueError("Service-level factor cannot be negative.")

    if simulation_days < 1:
        raise ValueError("Simulation duration must be at least one day.")

    # ---------------------------------------------------------
    # 2. Validate required columns
    # ---------------------------------------------------------

    required_sales = {
        "date",
        "store_id",
        "product_id",
        "units_sold",
    }

    required_inventory = {
        "store_id",
        "product_id",
        "on_hand",
        "lead_time_days",
        "unit_cost",
    }

    if not required_sales.issubset(sales.columns):
        missing = required_sales - set(sales.columns)
        raise ValueError(
            f"Sales data is missing columns: {sorted(missing)}"
        )

    if not required_inventory.issubset(inventory.columns):
        missing = required_inventory - set(inventory.columns)
        raise ValueError(
            f"Inventory data is missing columns: {sorted(missing)}"
        )

    # Make copies to preserve the original datasets.
    sales = sales.copy()
    inventory = inventory.copy()

    sales["date"] = pd.to_datetime(
        sales["date"],
        errors="raise",
    )

    sales["units_sold"] = pd.to_numeric(
        sales["units_sold"],
        errors="raise",
    )

    for column in ["on_hand", "lead_time_days", "unit_cost"]:
        inventory[column] = pd.to_numeric(
            inventory[column],
            errors="raise",
        )

    # ---------------------------------------------------------
    # 3. Validate data values
    # ---------------------------------------------------------

    if sales.empty:
        raise ValueError("Sales dataset is empty.")

    if inventory.empty:
        raise ValueError("Inventory dataset is empty.")

    if sales[
        ["date", "store_id", "product_id", "units_sold"]
    ].isna().any().any():
        raise ValueError("Sales data contains missing required values.")

    if inventory[
        [
            "store_id",
            "product_id",
            "on_hand",
            "lead_time_days",
            "unit_cost",
        ]
    ].isna().any().any():
        raise ValueError(
            "Inventory data contains missing required values."
        )

    if (sales["units_sold"] < 0).any():
        raise ValueError("Historical demand cannot be negative.")

    if (inventory["on_hand"] < 0).any():
        raise ValueError("On-hand inventory cannot be negative.")

    if (inventory["lead_time_days"] < 1).any():
        raise ValueError("Supplier lead time must be at least one day.")

    if (inventory["unit_cost"] < 0).any():
        raise ValueError("Unit cost cannot be negative.")

    if not np.isfinite(
        sales["units_sold"].to_numpy(dtype=float)
    ).all():
        raise ValueError("Historical demand must contain finite numbers.")

    if not np.isfinite(
        inventory[
            ["on_hand", "lead_time_days", "unit_cost"]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Inventory values must be finite numbers.")

    if not np.all(
        inventory["lead_time_days"]
        == np.floor(inventory["lead_time_days"])
    ):
        raise ValueError("Supplier lead time must use whole days.")

    if inventory.duplicated(
        ["store_id", "product_id"]
    ).any():
        raise ValueError(
            "Inventory must contain one row per store-product pair."
        )

    # ---------------------------------------------------------
    # 4. Prepare historical demand
    # ---------------------------------------------------------

    sales = sales.sort_values("date")

    daily_demand = (
        sales.groupby(
            ["store_id", "product_id", "date"],
            as_index=False,
        )["units_sold"]
        .sum()
    )

    demand_statistics = (
        daily_demand.groupby(
            ["store_id", "product_id"]
        )["units_sold"]
        .agg(
            mean_daily_demand="mean",
            std_daily_demand="std",
        )
        .reset_index()
    )

    demand_statistics["std_daily_demand"] = (
        demand_statistics["std_daily_demand"].fillna(0)
    )

    statistics_lookup = {
        (row.store_id, row.product_id): (
            float(row.mean_daily_demand),
            float(row.std_daily_demand),
        )
        for row in demand_statistics.itertuples(index=False)
    }

    demand_history = {}

    for key, group in daily_demand.groupby(
        ["store_id", "product_id"]
    ):
        demand_history[key] = group["units_sold"].to_numpy(
            dtype=int
        )

    # ---------------------------------------------------------
    # 5. Set the simulation period
    # ---------------------------------------------------------

    last_historical_date = sales["date"].max()

    simulation_start = (
        last_historical_date + pd.Timedelta(days=1)
    )

    simulation_dates = pd.date_range(
        start=simulation_start,
        periods=simulation_days,
        freq="D",
    )

    # Use a fixed seed so results can be reproduced.
    rng = np.random.default_rng(RANDOM_SEED)

    daily_records = []

    # ---------------------------------------------------------
    # 6. Simulate each store-product combination
    # ---------------------------------------------------------

    for item in inventory.itertuples(index=False):

        store_id = item.store_id
        product_id = item.product_id
        key = (store_id, product_id)

        if key not in statistics_lookup:
            raise ValueError(
                f"No historical sales found for {key}."
            )

        if key not in demand_history:
            raise ValueError(
                f"No daily demand history found for {key}."
            )

        mean_demand, std_demand = statistics_lookup[key]

        history = demand_history[key]

        lead_time = int(item.lead_time_days)
        unit_cost = float(item.unit_cost)

        # Start with the current stock from the inventory dataset.
        on_hand = int(item.on_hand)

        # A simple reorder-point policy is used.
        safety_stock = int(
            np.ceil(
                service_level_z
                * std_demand
                * np.sqrt(lead_time)
            )
        )

        reorder_point = int(
            np.ceil(
                mean_demand * lead_time + safety_stock
            )
        )

        target_stock = int(
            np.ceil(
                mean_demand
                * (lead_time + review_period_days)
                + safety_stock
            )
        )

        # Scheduled deliveries:
        # key = delivery date, value = quantity arriving.
        scheduled_deliveries = defaultdict(int)

        # Count outstanding order quantities.
        outstanding_order_qty = 0

        # Used to avoid placing duplicate orders on the same day.
        last_order_date = None

        for current_date in simulation_dates:

            # -------------------------------------------------
            # A. Receive orders scheduled for today
            # -------------------------------------------------

            delivery_qty = scheduled_deliveries.pop(
                current_date,
                0,
            )

            on_hand += delivery_qty

            outstanding_order_qty = max(
                0,
                outstanding_order_qty - delivery_qty,
            )

            # -------------------------------------------------
            # B. Generate synthetic customer demand
            # -------------------------------------------------
            # Resample historical daily demand to preserve
            # the general demand pattern and zero-demand days.

            simulated_demand = int(
                rng.choice(history)
            )

            # -------------------------------------------------
            # C. Fulfil demand from available inventory
            # -------------------------------------------------

            opening_stock = on_hand

            fulfilled_sales = min(
                on_hand,
                simulated_demand,
            )

            lost_sales = max(
                0,
                simulated_demand - fulfilled_sales,
            )

            on_hand -= fulfilled_sales

            # -------------------------------------------------
            # D. Check whether replenishment is needed
            # -------------------------------------------------

            inventory_position = (
                on_hand + outstanding_order_qty
            )

            order_qty = 0
            expected_delivery_date = pd.NaT

            if (
                inventory_position <= reorder_point
                and last_order_date != current_date
            ):

                # Order enough to reach the target stock level.
                order_qty = max(
                    0,
                    target_stock - inventory_position,
                )

                if order_qty > 0:

                    expected_delivery_date = (
                        current_date
                        + pd.Timedelta(days=lead_time)
                    )

                    scheduled_deliveries[
                        expected_delivery_date
                    ] += order_qty

                    outstanding_order_qty += order_qty

                    last_order_date = current_date

            # -------------------------------------------------
            # E. Record daily results
            # -------------------------------------------------

            daily_records.append(
                {
                    "date": current_date,
                    "store_id": store_id,
                    "product_id": product_id,
                    "opening_stock": opening_stock,
                    "customer_demand": simulated_demand,
                    "fulfilled_sales": fulfilled_sales,
                    "lost_sales": lost_sales,
                    "delivery_qty": delivery_qty,
                    "closing_stock": on_hand,
                    "reorder_point": reorder_point,
                    "target_stock": target_stock,
                    "order_qty": order_qty,
                    "outstanding_order_qty": (
                        outstanding_order_qty
                    ),
                    "expected_delivery_date": (
                        expected_delivery_date
                    ),
                    "estimated_order_cost": round(
                        order_qty * unit_cost,
                        2,
                    ),
                }
            )

    # ---------------------------------------------------------
    # 7. Create the daily simulation dataset
    # ---------------------------------------------------------

    daily_results = pd.DataFrame(daily_records)

    daily_results = daily_results.sort_values(
        ["date", "store_id", "product_id"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 8. Create a summary for every store-product pair
    # ---------------------------------------------------------

    summary = (
        daily_results.groupby(
            ["store_id", "product_id"],
            as_index=False,
        )
        .agg(
            total_customer_demand=("customer_demand", "sum"),
            total_fulfilled_sales=("fulfilled_sales", "sum"),
            total_lost_sales=("lost_sales", "sum"),
            total_delivered_units=("delivery_qty", "sum"),
            total_ordered_units=("order_qty", "sum"),
            total_order_cost=("estimated_order_cost", "sum"),
            average_closing_stock=("closing_stock", "mean"),
            minimum_closing_stock=("closing_stock", "min"),
            final_closing_stock=("closing_stock", "last"),
            days_with_stockout=(
                "lost_sales",
                lambda values: int((values > 0).sum()),
            ),
            reorder_events=("order_qty", lambda values: int((values > 0).sum())),
        )
    )

    summary["fulfilment_rate_pct"] = np.where(
        summary["total_customer_demand"] > 0,
        (
            summary["total_fulfilled_sales"]
            / summary["total_customer_demand"]
            * 100
        ),
        100.0,
    )

    summary["lost_sales_rate_pct"] = np.where(
        summary["total_customer_demand"] > 0,
        (
            summary["total_lost_sales"]
            / summary["total_customer_demand"]
            * 100
        ),
        0.0,
    )

    summary["total_order_cost"] = summary[
        "total_order_cost"
    ].round(2)

    summary["average_closing_stock"] = summary[
        "average_closing_stock"
    ].round(2)

    summary["fulfilment_rate_pct"] = summary[
        "fulfilment_rate_pct"
    ].round(2)

    summary["lost_sales_rate_pct"] = summary[
        "lost_sales_rate_pct"
    ].round(2)

    return daily_results, summary


def run_inventory_simulation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the synthetic datasets, run the inventory simulation,
    save the outputs, and print a summary.
    """

    sales_path = (
        PROCESSED_DATA_DIR / "cleaned_sales.csv"
    )

    inventory_path = (
        SIMULATED_DATA_DIR / "inventory.csv"
    )

    for path in [sales_path, inventory_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Generate and preprocess the datasets first."
            )

    sales = pd.read_csv(
        sales_path,
        parse_dates=["date"],
    )

    inventory = pd.read_csv(
        inventory_path,
    )

    daily_results, summary = simulate_inventory(
        sales=sales,
        inventory=inventory,
        review_period_days=7,
        service_level_z=1.65,
        simulation_days=30,
    )

    # ---------------------------------------------------------
    # Save simulation outputs
    # ---------------------------------------------------------

    INVENTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_output_path = (
        INVENTORY_DIR / "daily_inventory_simulation.csv"
    )

    summary_output_path = (
        INVENTORY_DIR / "inventory_simulation_summary.csv"
    )

    daily_results.to_csv(
        daily_output_path,
        index=False,
    )

    summary.to_csv(
        summary_output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Display the results
    # ---------------------------------------------------------

    total_demand = int(
        daily_results["customer_demand"].sum()
    )

    total_fulfilled = int(
        daily_results["fulfilled_sales"].sum()
    )

    total_lost = int(
        daily_results["lost_sales"].sum()
    )

    total_ordered = int(
        daily_results["order_qty"].sum()
    )

    total_delivered = int(
        daily_results["delivery_qty"].sum()
    )

    total_order_cost = float(
        daily_results["estimated_order_cost"].sum()
    )

    overall_fulfilment_rate = (
        total_fulfilled / total_demand * 100
        if total_demand > 0
        else 100.0
    )

    print("=" * 70)
    print("RETAIL INVENTORY SIMULATION RESULTS")
    print("=" * 70)

    print("Simulation type: Synthetic and educational")
    print(
        f"Simulation days: "
        f"{daily_results['date'].nunique()}"
    )

    print(
        f"Store-product combinations: "
        f"{summary.shape[0]}"
    )

    print(f"Total customer demand: {total_demand:,}")
    print(f"Total fulfilled sales: {total_fulfilled:,}")
    print(f"Total lost sales: {total_lost:,}")
    print(f"Total units ordered: {total_ordered:,}")
    print(f"Total units delivered: {total_delivered:,}")

    print(
        "Overall fulfilment rate: "
        f"{overall_fulfilment_rate:.2f}%"
    )

    print(
        "Estimated order cost: "
        f"{total_order_cost:,.2f}"
    )

    print("\nINVENTORY SIMULATION SUMMARY")
    print("-" * 70)

    print(
        summary[
            [
                "store_id",
                "product_id",
                "total_customer_demand",
                "total_fulfilled_sales",
                "total_lost_sales",
                "days_with_stockout",
                "final_closing_stock",
                "reorder_events",
            ]
        ].to_string(index=False)
    )

    print("\nSaved files:")
    print(daily_output_path)
    print(summary_output_path)

    print(
        "\nNote: Simulation results depend on synthetic demand, "
        "starting inventory, and the selected reorder policy."
    )

    return daily_results, summary


if __name__ == "__main__":
    run_inventory_simulation()
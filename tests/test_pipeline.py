"""Automated validation tests for the retail forecasting project."""

from pathlib import Path

import numpy as np
import pandas as pd


# Locate the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_csv(relative_path):
    """Load a CSV file from the project directory."""
    file_path = PROJECT_ROOT / relative_path

    assert file_path.exists(), f"Required file is missing: {relative_path}"

    return pd.read_csv(file_path)


def test_cleaned_sales_data():
    """Check the quality of the cleaned sales dataset."""
    sales = load_csv("data/processed/cleaned_sales.csv")

    required_columns = {
        "date",
        "store_id",
        "product_id",
        "units_sold",
        "unit_price",
        "revenue",
    }

    assert required_columns.issubset(sales.columns)
    assert not sales.empty
    assert not sales[list(required_columns)].isnull().any().any()
    assert (sales["units_sold"] >= 0).all()
    assert (sales["unit_price"] > 0).all()
    assert (sales["revenue"] >= 0).all()


def test_sales_forecast():
    """Check the generated sales forecasts."""
    forecast = load_csv("outputs/forecasts/sales_forecast.csv")

    required_columns = {
        "date",
        "store_id",
        "product_id",
    }

    assert required_columns.issubset(forecast.columns)
    assert not forecast.empty

    prediction_columns = [
        column
        for column in forecast.columns
        if "forecast" in column.lower()
        or "prediction" in column.lower()
        or "predicted" in column.lower()
    ]

    assert prediction_columns, (
        "No forecast or prediction column was found. "
        "Check the column names in sales_forecast.csv."
    )

    for column in prediction_columns:
        assert pd.to_numeric(
            forecast[column], errors="coerce"
        ).notna().all(), f"Invalid values in {column}"

        assert (forecast[column] >= 0).all(), (
            f"Negative forecast values found in {column}"
        )

    forecast["date"] = pd.to_datetime(forecast["date"])

    assert forecast["date"].nunique() == 14
    assert not forecast.duplicated(
        ["date", "store_id", "product_id"]
    ).any()


def test_forecast_metrics():
    """Check that model evaluation metrics are valid."""
    metrics = load_csv("outputs/metrics/forecast_metrics.csv")

    assert not metrics.empty

    numeric_columns = [
        column
        for column in ["MAE", "RMSE"]
        if column in metrics.columns
    ]

    assert numeric_columns, "MAE and RMSE columns were not found."

    for column in numeric_columns:
        values = pd.to_numeric(metrics[column], errors="coerce")

        assert values.notna().all(), f"Invalid values in {column}"
        assert np.isfinite(values).all()
        assert (values >= 0).all()


def test_inventory_recommendations():
    """Check the inventory optimization recommendations."""
    inventory = load_csv(
        "outputs/inventory/inventory_recommendations.csv"
    )

    assert not inventory.empty
    assert {"store_id", "product_id"}.issubset(inventory.columns)

    assert not inventory.duplicated(
        ["store_id", "product_id"]
    ).any()

    quantity_columns = [
        column
        for column in inventory.columns
        if "order" in column.lower()
        and ("qty" in column.lower() or "quantity" in column.lower())
    ]

    assert quantity_columns, (
        "No suggested order quantity column was found. "
        "Check inventory_recommendations.csv."
    )

    for column in quantity_columns:
        values = pd.to_numeric(inventory[column], errors="coerce")

        assert values.notna().all(), f"Invalid values in {column}"
        assert (values >= 0).all(), (
            f"Negative order quantities found in {column}"
        )


def test_inventory_simulation():
    """Check the daily inventory simulation results."""
    simulation = load_csv(
        "outputs/inventory/daily_inventory_simulation.csv"
    )

    required_columns = {
        "date",
        "store_id",
        "product_id",
        "customer_demand",
        "fulfilled_sales",
        "lost_sales",
    }

    assert required_columns.issubset(simulation.columns)
    assert not simulation.empty

    numeric_columns = [
        "customer_demand",
        "fulfilled_sales",
        "lost_sales",
    ]

    for column in numeric_columns:
        simulation[column] = pd.to_numeric(
            simulation[column], errors="coerce"
        )

        assert simulation[column].notna().all()
        assert (simulation[column] >= 0).all()

    assert np.allclose(
        simulation["customer_demand"],
        simulation["fulfilled_sales"] + simulation["lost_sales"],
    )

    simulation["date"] = pd.to_datetime(simulation["date"])

    assert simulation["date"].nunique() == 30


def test_inventory_simulation_summary():
    """Check the overall inventory simulation summary."""
    summary = load_csv(
        "outputs/inventory/inventory_simulation_summary.csv"
    )

    assert not summary.empty
    assert len(summary.columns) >= 2
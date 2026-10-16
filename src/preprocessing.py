"""Validate and clean retail sales data."""

import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    SIMULATED_DATA_DIR,
)


REQUIRED_COLUMNS = [
    "date",
    "store_id",
    "product_id",
    "product_name",
    "category",
    "units_sold",
    "unit_price",
    "revenue",
]


def clean_sales_data(
    sales: pd.DataFrame,
    save_output: bool = True,
) -> pd.DataFrame:
    """
    Validate and clean retail sales records.

    Rules:
    - Require all essential columns.
    - Convert dates and numeric fields to appropriate types.
    - Remove exact duplicate rows.
    - Remove rows with missing essential values.
    - Remove records with invalid quantities or prices.
    - Recalculate revenue from quantity and unit price.
    """

    missing_columns = set(REQUIRED_COLUMNS) - set(sales.columns)

    if missing_columns:
        raise ValueError(
            f"Sales data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    cleaned = sales[REQUIRED_COLUMNS].copy()

    original_count = len(cleaned)

    # Remove exact duplicate records.
    cleaned = cleaned.drop_duplicates()

    # Convert data types.
    cleaned["date"] = pd.to_datetime(
        cleaned["date"],
        errors="coerce",
    )

    numeric_columns = [
        "units_sold",
        "unit_price",
        "revenue",
    ]

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    # Remove records with missing essential values.
    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS)

    # Sales quantities must be non-negative whole numbers.
    valid_quantity = (
        (cleaned["units_sold"] >= 0)
        & (cleaned["units_sold"] % 1 == 0)
    )

    # Prices must be positive.
    valid_price = cleaned["unit_price"] > 0

    cleaned = cleaned.loc[
        valid_quantity & valid_price
    ].copy()

    # Convert validated quantities to integers.
    cleaned["units_sold"] = cleaned["units_sold"].astype(int)

    # Recalculate revenue to maintain consistency.
    cleaned["revenue"] = (
        cleaned["units_sold"] * cleaned["unit_price"]
    ).round(2)

    # Sort chronologically for later forecasting.
    cleaned = cleaned.sort_values(
        ["date", "store_id", "product_id"]
    ).reset_index(drop=True)

    if save_output:
        PROCESSED_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            PROCESSED_DATA_DIR / "cleaned_sales.csv"
        )

        cleaned.to_csv(output_path, index=False)

    removed_count = original_count - len(cleaned)

    print("=" * 50)
    print("SALES DATA CLEANING REPORT")
    print("=" * 50)
    print(f"Original records: {original_count:,}")
    print(f"Cleaned records: {len(cleaned):,}")
    print(f"Records removed: {removed_count:,}")
    print(
        "Missing values remaining:",
        cleaned.isna().sum().sum(),
    )
    print(
        "Negative sales quantities:",
        (cleaned["units_sold"] < 0).sum(),
    )
    print(
        "Non-positive unit prices:",
        (cleaned["unit_price"] <= 0).sum(),
    )

    if save_output:
        print(f"Saved cleaned data to: {output_path}")

    return cleaned


def load_and_clean_sales() -> pd.DataFrame:
    """Load the synthetic sales CSV and clean it."""
    input_path = SIMULATED_DATA_DIR / "sales_history.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Sales data not found at {input_path}. "
            "Run python -m src.data_simulator first."
        )

    sales = pd.read_csv(input_path)

    return clean_sales_data(sales)


if __name__ == "__main__":
    load_and_clean_sales()
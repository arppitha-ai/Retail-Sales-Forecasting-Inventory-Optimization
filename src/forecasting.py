"""
Retail Sales Forecasting Module.

Features:
- Historical demand feature engineering
- Consistent store and product encoding
- Naive forecasting baseline
- 7-day moving average baseline
- Random Forest forecasting model
- Chronological model evaluation
- Recursive future forecasting
- MAE and RMSE evaluation metrics
- CSV output generation

IMPORTANT:
This project uses synthetic retail data.
Forecast performance does not establish real-world accuracy.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import (
    FORECAST_DIR,
    FORECAST_HORIZON,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TEST_SIZE_DAYS,
)


# =============================================================
# CONFIGURATION
# =============================================================

FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "day_of_week",
    "month",
    "product_code",
    "store_code",
]


# =============================================================
# DATA LOADING AND VALIDATION
# =============================================================

def load_sales() -> pd.DataFrame:
    """Load and validate cleaned historical sales."""

    path = PROCESSED_DATA_DIR / "cleaned_sales.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned sales data not found: {path}\n"
            "Run python -m src.preprocessing first."
        )

    sales = pd.read_csv(path, parse_dates=["date"])

    required_columns = {
        "date",
        "store_id",
        "product_id",
        "product_name",
        "units_sold",
    }

    missing_columns = required_columns - set(sales.columns)

    if missing_columns:
        raise ValueError(
            f"Sales data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if sales.empty:
        raise ValueError("The sales dataset is empty.")

    if sales[list(required_columns)].isna().any().any():
        raise ValueError(
            "Sales data contains missing values in required columns."
        )

    sales["units_sold"] = pd.to_numeric(
        sales["units_sold"],
        errors="raise",
    )

    if not np.isfinite(
        sales["units_sold"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Sales quantities must contain finite numbers."
        )

    if (sales["units_sold"] < 0).any():
        raise ValueError(
            "Historical sales quantities cannot be negative."
        )

    sales["store_id"] = sales["store_id"].astype(str)
    sales["product_id"] = sales["product_id"].astype(str)

    sales = sales.sort_values(
        ["store_id", "product_id", "date"]
    ).reset_index(drop=True)

    # Require exactly one observation per store-product-date.
    duplicate_rows = sales.duplicated(
        ["store_id", "product_id", "date"]
    )

    if duplicate_rows.any():
        raise ValueError(
            "Duplicate store-product-date records were found. "
            "Clean the data before forecasting."
        )

    return sales


# =============================================================
# CONSISTENT CATEGORY ENCODING
# =============================================================

def create_category_mappings(
    sales: pd.DataFrame,
) -> tuple[dict, dict]:
    """
    Create stable numerical codes for stores and products.

    The mappings are created from the complete set of known
    store and product IDs and reused during future forecasting.

    This prevents individual store-product groups from being
    incorrectly assigned the same category code.
    """

    product_ids = sorted(
        sales["product_id"].astype(str).unique()
    )

    store_ids = sorted(
        sales["store_id"].astype(str).unique()
    )

    product_mapping = {
        product_id: index
        for index, product_id in enumerate(product_ids)
    }

    store_mapping = {
        store_id: index
        for index, store_id in enumerate(store_ids)
    }

    return product_mapping, store_mapping


# =============================================================
# FEATURE ENGINEERING
# =============================================================

def add_features(
    sales: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add historical demand, calendar, store, and product features.

    Lag and rolling features use previous observations only.
    """

    result = sales.copy()

    result["date"] = pd.to_datetime(result["date"])

    result["store_id"] = result["store_id"].astype(str)
    result["product_id"] = result["product_id"].astype(str)

    result = result.sort_values(
        ["store_id", "product_id", "date"]
    ).reset_index(drop=True)

    # Calendar features.
    result["day_of_week"] = result["date"].dt.dayofweek
    result["month"] = result["date"].dt.month

    # Stable category mappings.
    product_mapping, store_mapping = (
        create_category_mappings(result)
    )

    result["product_code"] = (
        result["product_id"].map(product_mapping)
    )

    result["store_code"] = (
        result["store_id"].map(store_mapping)
    )

    # Group historical demand by store and product.
    grouped = result.groupby(
        ["store_id", "product_id"],
        sort=False,
    )["units_sold"]

    # Previous day's demand.
    result["lag_1"] = grouped.shift(1)

    # Demand seven days earlier.
    result["lag_7"] = grouped.shift(7)

    # Mean of the previous seven days, excluding the current day.
    result["rolling_mean_7"] = grouped.transform(
        lambda values: values.shift(1).rolling(
            window=7,
            min_periods=7,
        ).mean()
    )

    return result


# =============================================================
# MODEL EVALUATION METRICS
# =============================================================

def calculate_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
) -> dict:
    """Calculate Mean Absolute Error and Root Mean Squared Error."""

    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)

    if len(actual_values) != len(predicted_values):
        raise ValueError(
            "Actual and predicted arrays must have equal lengths."
        )

    if len(actual_values) == 0:
        raise ValueError(
            "Cannot calculate metrics for empty arrays."
        )

    return {
        "MAE": float(
            mean_absolute_error(
                actual_values,
                predicted_values,
            )
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    actual_values,
                    predicted_values,
                )
            )
        ),
    }


# =============================================================
# MODEL TRAINING AND CHRONOLOGICAL EVALUATION
# =============================================================

def evaluate_models(
    features: pd.DataFrame,
    test_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, RandomForestRegressor]:
    """
    Compare three forecasting methods.

    Evaluation:
    - Naive: previous day's observed demand
    - 7-Day Moving Average: mean of previous seven days
    - Random Forest: regression using engineered features

    The test uses rolling one-day-ahead evaluation. Actual
    historical observations from preceding test dates are
    available when calculating subsequent test-date features.
    """

    usable = features.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    train = usable[
        usable["date"] < test_start
    ].copy()

    test = usable[
        usable["date"] >= test_start
    ].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Insufficient data for chronological model evaluation."
        )

    X_train = train[FEATURE_COLUMNS]
    y_train = train["units_sold"]

    X_test = test[FEATURE_COLUMNS]
    y_test = test["units_sold"]

    # ---------------------------------------------------------
    # Model 1: Naive baseline
    # ---------------------------------------------------------

    naive_predictions = test["lag_1"].to_numpy(
        dtype=float
    )

    # ---------------------------------------------------------
    # Model 2: Seven-day moving average baseline
    # ---------------------------------------------------------

    moving_average_predictions = (
        test["rolling_mean_7"].to_numpy(dtype=float)
    )

    # ---------------------------------------------------------
    # Model 3: Random Forest
    # ---------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    random_forest_predictions = model.predict(X_test)

    # Demand cannot be negative.
    naive_predictions = np.maximum(
        naive_predictions,
        0,
    )

    moving_average_predictions = np.maximum(
        moving_average_predictions,
        0,
    )

    random_forest_predictions = np.maximum(
        random_forest_predictions,
        0,
    )

    # ---------------------------------------------------------
    # Build the test predictions table
    # ---------------------------------------------------------

    prediction_parts = []

    model_predictions = [
        ("Naive", naive_predictions),
        (
            "7-Day Moving Average",
            moving_average_predictions,
        ),
        (
            "Random Forest",
            random_forest_predictions,
        ),
    ]

    for model_name, predictions in model_predictions:

        part = test[
            [
                "date",
                "store_id",
                "product_id",
                "product_name",
                "units_sold",
            ]
        ].copy()

        part["model"] = model_name

        part["actual_units"] = part.pop("units_sold")

        part["predicted_units"] = predictions

        prediction_parts.append(part)

    test_predictions = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Calculate model metrics
    # ---------------------------------------------------------

    metric_records = []

    for model_name, predictions in model_predictions:

        scores = calculate_metrics(
            y_test,
            predictions,
        )

        metric_records.append(
            {
                "model": model_name,
                "MAE": scores["MAE"],
                "RMSE": scores["RMSE"],
                "test_start": test["date"].min().date(),
                "test_end": test["date"].max().date(),
                "test_records": len(test),
            }
        )

    metrics = pd.DataFrame(
        metric_records
    ).sort_values(
        "MAE"
    ).reset_index(drop=True)

    return test_predictions, metrics, model


# =============================================================
# FUTURE FORECAST GENERATION
# =============================================================

def generate_future_forecasts(
    sales: pd.DataFrame,
    model: RandomForestRegressor,
) -> pd.DataFrame:
    """
    Generate recursive forecasts for each store-product pair.

    Stable product and store mappings are recreated from the
    complete sales dataset, matching the training encoding.

    Predictions are fed into subsequent forecasting steps.
    """

    if FORECAST_HORIZON < 1:
        raise ValueError(
            "Forecast horizon must be at least one day."
        )

    sales = sales.copy()

    sales["date"] = pd.to_datetime(sales["date"])

    sales["store_id"] = sales["store_id"].astype(str)
    sales["product_id"] = sales["product_id"].astype(str)

    sales = sales.sort_values(
        ["store_id", "product_id", "date"]
    ).reset_index(drop=True)

    product_mapping, store_mapping = (
        create_category_mappings(sales)
    )

    forecasts = []

    # Each group represents one store-product combination.
    for (store_id, product_id), group in sales.groupby(
        ["store_id", "product_id"],
        sort=True,
    ):

        group = group.sort_values("date").copy()

        product_name = group["product_name"].iloc[-1]

        history = (
            group["units_sold"]
            .astype(float)
            .tolist()
        )

        last_date = group["date"].max()

        # IMPORTANT:
        # These codes come from the complete dataset mappings,
        # not from category codes generated inside an individual
        # store-product group.
        product_code = product_mapping[product_id]
        store_code = store_mapping[store_id]

        if len(history) < 7:
            raise ValueError(
                f"At least seven observations are required for "
                f"store {store_id}, product {product_id}."
            )

        # Generate forecasts one day at a time.
        for step in range(
            1,
            FORECAST_HORIZON + 1,
        ):

            forecast_date = (
                last_date + pd.Timedelta(days=step)
            )

            last_seven = history[-7:]

            feature_row = pd.DataFrame(
                [
                    {
                        "lag_1": history[-1],
                        "lag_7": history[-7],
                        "rolling_mean_7": float(
                            np.mean(last_seven)
                        ),
                        "day_of_week": (
                            forecast_date.dayofweek
                        ),
                        "month": forecast_date.month,
                        "product_code": product_code,
                        "store_code": store_code,
                    }
                ],
                columns=FEATURE_COLUMNS,
            )

            prediction = float(
                model.predict(feature_row)[0]
            )

            # Demand cannot be negative.
            prediction = max(
                prediction,
                0.0,
            )

            # Round up to whole units for inventory planning.
            planning_units = int(
                np.ceil(prediction)
            )

            forecasts.append(
                {
                    "date": forecast_date,
                    "store_id": store_id,
                    "product_id": product_id,
                    "product_name": product_name,
                    "predicted_units": round(
                        prediction,
                        2,
                    ),
                    "planning_units": planning_units,
                }
            )

            # Append the prediction for the next recursive step.
            history.append(prediction)

    result = pd.DataFrame(forecasts)

    if result.empty:
        raise ValueError(
            "No future forecasts were generated."
        )

    return result.sort_values(
        ["store_id", "product_id", "date"]
    ).reset_index(drop=True)


# =============================================================
# MAIN FORECASTING PIPELINE
# =============================================================

def run_forecasting() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the complete forecasting workflow.

    Outputs:
    - outputs/forecasts/test_predictions.csv
    - outputs/forecasts/sales_forecast.csv
    - outputs/metrics/forecast_metrics.csv
    """

    FORECAST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 1. Load historical data
    # ---------------------------------------------------------

    sales = load_sales()

    unique_dates = sorted(
        sales["date"].unique()
    )

    if len(unique_dates) <= TEST_SIZE_DAYS + 7:
        raise ValueError(
            "Not enough historical dates for forecasting. "
            "More than TEST_SIZE_DAYS + 7 dates are required."
        )

    test_start = pd.Timestamp(
        unique_dates[-TEST_SIZE_DAYS]
    )

    # ---------------------------------------------------------
    # 2. Generate model features
    # ---------------------------------------------------------

    features = add_features(sales)

    # ---------------------------------------------------------
    # 3. Evaluate models
    # ---------------------------------------------------------

    test_predictions, metrics, model = evaluate_models(
        features,
        test_start,
    )

    # ---------------------------------------------------------
    # 4. Generate future forecasts
    # ---------------------------------------------------------

    future_forecasts = generate_future_forecasts(
        sales,
        model,
    )

    # ---------------------------------------------------------
    # 5. Save outputs
    # ---------------------------------------------------------

    predictions_path = (
        FORECAST_DIR / "test_predictions.csv"
    )

    metrics_path = (
        METRICS_DIR / "forecast_metrics.csv"
    )

    forecasts_path = (
        FORECAST_DIR / "sales_forecast.csv"
    )

    test_predictions.to_csv(
        predictions_path,
        index=False,
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    future_forecasts.to_csv(
        forecasts_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 6. Display results
    # ---------------------------------------------------------

    print("=" * 65)
    print("RETAIL SALES FORECASTING RESULTS")
    print("=" * 65)

    print(
        "Evaluation method: "
        "Rolling one-day-ahead chronological evaluation"
    )

    print(
        f"Training data ends before: {test_start.date()}"
    )

    print(
        f"Test period starts: {test_start.date()}"
    )

    print(
        f"Forecast horizon: {FORECAST_HORIZON} days"
    )

    print(
        "Test observations per model: "
        f"{len(test_predictions) // 3:,}"
    )

    print("\nMODEL EVALUATION")
    print("-" * 65)

    print(
        metrics[
            ["model", "MAE", "RMSE"]
        ].to_string(
            index=False,
            formatters={
                "MAE": "{:.3f}".format,
                "RMSE": "{:.3f}".format,
            },
        )
    )

    best_model = metrics.iloc[0]["model"]

    print(
        f"\nBest model by MAE on the synthetic test period: "
        f"{best_model}"
    )

    print("\nFUTURE FORECAST SAMPLE")
    print("-" * 65)

    print(
        future_forecasts.head(10).to_string(
            index=False
        )
    )

    print("\nFORECAST VALIDATION")
    print("-" * 65)

    expected_records = (
        FORECAST_HORIZON
        * sales[["store_id", "product_id"]]
        .drop_duplicates()
        .shape[0]
    )

    print(
        f"Expected forecast records: {expected_records}"
    )

    print(
        f"Actual forecast records: {len(future_forecasts)}"
    )

    if len(future_forecasts) != expected_records:
        raise ValueError(
            "The number of generated forecast records "
            "does not match the expected horizon."
        )

    print(
        "Forecast start date: "
        f"{future_forecasts['date'].min().date()}"
    )

    print(
        "Forecast end date: "
        f"{future_forecasts['date'].max().date()}"
    )

    print("\nSAVED OUTPUT FILES")
    print("-" * 65)

    print(f"Test predictions: {predictions_path}")
    print(f"Evaluation metrics: {metrics_path}")
    print(f"Future forecasts: {forecasts_path}")

    print("\nForecasting completed successfully.")

    print(
        "Reminder: all forecasts and evaluation results "
        "are based on synthetic project data."
    )

    return future_forecasts, metrics


if __name__ == "__main__":
    run_forecasting()
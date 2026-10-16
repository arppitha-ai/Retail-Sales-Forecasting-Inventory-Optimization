"""
Retail Sales Forecasting & Inventory Optimization Dashboard

Interactive dashboard for exploring:
- Historical retail sales
- Sales forecasting
- Forecast model evaluation
- Inventory recommendations
- Inventory simulation
- Downloadable project data

All data is synthetic and intended for educational purposes.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Retail Analytics | Forecasting & Inventory",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEANED_SALES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "cleaned_sales.csv"
)

FORECAST_PATH = (
    PROJECT_ROOT / "outputs" / "forecasts" / "sales_forecast.csv"
)

METRICS_PATH = (
    PROJECT_ROOT / "outputs" / "metrics" / "forecast_metrics.csv"
)

INVENTORY_PATH = (
    PROJECT_ROOT / "outputs" / "inventory"
    / "inventory_recommendations.csv"
)

SIMULATION_DAILY_PATH = (
    PROJECT_ROOT / "outputs" / "inventory"
    / "daily_inventory_simulation.csv"
)

SIMULATION_SUMMARY_PATH = (
    PROJECT_ROOT / "outputs" / "inventory"
    / "inventory_simulation_summary.csv"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f7fb;
    }

    [data-testid="stSidebar"] {
        background-color: #eaf1fb;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #123b70;
        margin-bottom: 0.15rem;
    }

    .subtitle {
        color: #52657d;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }

    .section-title {
        color: #123b70;
        font-weight: 700;
    }

    .notice {
        padding: 12px 16px;
        background-color: #e8f2ff;
        border-left: 4px solid #2475c7;
        border-radius: 6px;
        color: #183d66;
        margin-bottom: 16px;
    }

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #dce5f1;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 2px 7px rgba(20, 50, 90, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #52657d;
    }

    h1, h2, h3 {
        color: #123b70;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_csv(path_string: str) -> pd.DataFrame | None:
    """Load a CSV file if it exists and can be read."""

    path = Path(path_string)

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None


sales = load_csv(str(CLEANED_SALES_PATH))
forecasts = load_csv(str(FORECAST_PATH))
metrics = load_csv(str(METRICS_PATH))
inventory = load_csv(str(INVENTORY_PATH))
simulation_daily = load_csv(str(SIMULATION_DAILY_PATH))
simulation_summary = load_csv(str(SIMULATION_SUMMARY_PATH))


# ============================================================
# GENERAL HELPERS
# ============================================================

def money(value: float) -> str:
    """Format a monetary value for display."""

    return f"{value:,.2f}"


def show_missing_file_message(
    filename: str,
    command: str,
) -> None:
    """Explain how to generate a missing project output."""

    st.warning(
        f"Required data file `{filename}` was not found. "
        f"Run `{command}` in your project terminal, then "
        "refresh this dashboard."
    )


def prepare_date_column(
    dataframe: pd.DataFrame,
    column: str = "date",
) -> pd.DataFrame:
    """Return a copy with a parsed date column."""

    result = dataframe.copy()
    result[column] = pd.to_datetime(
        result[column],
        errors="coerce",
    )

    return result.dropna(subset=[column])


def chart_layout(fig, height: int = 380):
    """Apply consistent layout styling to Plotly figures."""

    fig.update_layout(
        height=height,
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text="",
        hovermode="x unified",
    )

    return fig


def safe_download(
    dataframe: pd.DataFrame,
    filename: str,
    label: str,
) -> None:
    """Display a CSV download button."""

    st.download_button(
        label=label,
        data=dataframe.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📊 Retail Analytics")

st.sidebar.caption(
    "Sales Forecasting & Inventory Optimization"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Sales Analysis",
        "Forecasting",
        "Inventory Optimization",
        "Inventory Simulation",
        "Data Explorer",
    ],
)

st.sidebar.divider()

st.sidebar.markdown("### Project Information")

st.sidebar.write("**Data:** Synthetic retail dataset")
st.sidebar.write("**Forecast model:** Random Forest")
st.sidebar.write("**Stores:** 3")
st.sidebar.write("**Products:** 8")

st.sidebar.caption(
    "Educational portfolio project. "
    "Results are not real business performance."
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Retail Sales Forecasting '
    '& Inventory Optimization</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Explore sales patterns, forecast demand, and evaluate '
    'inventory planning through an interactive dashboard.'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="notice">'
    '<strong>Simulation notice:</strong> All data and '
    'results shown in this project are synthetic. '
    'Inventory recommendations and forecasts are estimates '
    'for educational demonstration.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.subheader("Business Overview")

    if sales is None:
        show_missing_file_message(
            "data/processed/cleaned_sales.csv",
            "python -m src.preprocessing",
        )

    else:
        sales = prepare_date_column(sales)

        required = {
            "date",
            "store_id",
            "product_id",
            "units_sold",
            "revenue",
        }

        if not required.issubset(sales.columns):
            st.error(
                "The cleaned sales file does not contain "
                "all required columns."
            )
            st.stop()

        sales["units_sold"] = pd.to_numeric(
            sales["units_sold"],
            errors="coerce",
        ).fillna(0)

        sales["revenue"] = pd.to_numeric(
            sales["revenue"],
            errors="coerce",
        ).fillna(0)

        total_units = int(sales["units_sold"].sum())
        total_revenue = float(sales["revenue"].sum())
        store_count = sales["store_id"].nunique()
        product_count = sales["product_id"].nunique()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Units Sold",
            f"{total_units:,}",
        )

        col2.metric(
            "Total Simulated Revenue",
            money(total_revenue),
        )

        col3.metric(
            "Stores",
            store_count,
        )

        col4.metric(
            "Products",
            product_count,
        )

        st.divider()

        # Daily sales trend.
        daily_sales = (
            sales.groupby("date", as_index=False)
            .agg(
                units_sold=("units_sold", "sum"),
                revenue=("revenue", "sum"),
            )
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Daily Sales Trend")

            fig = px.line(
                daily_sales,
                x="date",
                y="units_sold",
                markers=True,
                title="Units Sold Over Time",
                labels={
                    "date": "Date",
                    "units_sold": "Units Sold",
                },
            )

            fig.update_traces(
                line=dict(color="#2475c7", width=2)
            )

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True,
            )

        with col2:
            st.subheader("Revenue Trend")

            fig = px.area(
                daily_sales,
                x="date",
                y="revenue",
                title="Simulated Daily Revenue",
                labels={
                    "date": "Date",
                    "revenue": "Revenue",
                },
            )

            fig.update_traces(
                line_color="#168f80",
                fillcolor="rgba(22,143,128,0.20)",
            )

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True,
            )

        st.subheader("Top Products by Units Sold")

        top_products = (
            sales.groupby(
                ["product_id", "product_name"],
                as_index=False,
            )
            .agg(units_sold=("units_sold", "sum"))
            .sort_values("units_sold", ascending=False)
        )

        fig = px.bar(
            top_products,
            x="product_name",
            y="units_sold",
            color="units_sold",
            color_continuous_scale="Blues",
            title="Product Demand Ranking",
            labels={
                "product_name": "Product",
                "units_sold": "Units Sold",
            },
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True,
        )

        safe_download(
            sales,
            "cleaned_sales.csv",
            "Download Cleaned Sales Data",
        )

    # Model summary.
    st.divider()
    st.subheader("Forecast Model Performance")

    if metrics is None:
        show_missing_file_message(
            "outputs/metrics/forecast_metrics.csv",
            "python -m src.forecasting",
        )

    elif {"model", "MAE", "RMSE"}.issubset(metrics.columns):

        metrics["MAE"] = pd.to_numeric(
            metrics["MAE"],
            errors="coerce",
        )

        metrics["RMSE"] = pd.to_numeric(
            metrics["RMSE"],
            errors="coerce",
        )

        metrics = metrics.dropna(
            subset=["MAE", "RMSE"]
        )

        if not metrics.empty:

            best = metrics.sort_values("MAE").iloc[0]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Best Model by MAE",
                str(best["model"]),
            )

            col2.metric(
                "Best MAE",
                f"{best['MAE']:.3f}",
            )

            col3.metric(
                "Best RMSE",
                f"{best['RMSE']:.3f}",
            )

            fig = px.bar(
                metrics.sort_values("MAE"),
                x="model",
                y="MAE",
                color="model",
                title="Model Comparison: MAE",
                labels={
                    "model": "Model",
                    "MAE": "Mean Absolute Error",
                },
            )

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True,
            )


# ============================================================
# SALES ANALYSIS PAGE
# ============================================================

elif page == "Sales Analysis":

    st.subheader("Sales Analysis")

    if sales is None:
        show_missing_file_message(
            "data/processed/cleaned_sales.csv",
            "python -m src.preprocessing",
        )

    else:
        sales = prepare_date_column(sales)

        sales["units_sold"] = pd.to_numeric(
            sales["units_sold"],
            errors="coerce",
        ).fillna(0)

        sales["revenue"] = pd.to_numeric(
            sales["revenue"],
            errors="coerce",
        ).fillna(0)

        st.sidebar.subheader("Sales Filters")

        stores = sorted(
            sales["store_id"].dropna().unique().tolist()
        )

        selected_stores = st.sidebar.multiselect(
            "Select Stores",
            stores,
            default=stores,
        )

        products = sorted(
            sales["product_id"].dropna().unique().tolist()
        )

        selected_products = st.sidebar.multiselect(
            "Select Products",
            products,
            default=products,
        )

        filtered = sales[
            sales["store_id"].isin(selected_stores)
            & sales["product_id"].isin(selected_products)
        ].copy()

        if filtered.empty:
            st.info(
                "No sales records match the selected filters."
            )

        else:
            daily = (
                filtered.groupby("date", as_index=False)
                .agg(
                    units_sold=("units_sold", "sum"),
                    revenue=("revenue", "sum"),
                )
            )

            col1, col2 = st.columns(2)

            with col1:
                fig = px.line(
                    daily,
                    x="date",
                    y="units_sold",
                    markers=True,
                    title="Filtered Daily Sales",
                )

                st.plotly_chart(
                    chart_layout(fig),
                    use_container_width=True,
                )

            with col2:
                fig = px.line(
                    daily,
                    x="date",
                    y="revenue",
                    title="Filtered Revenue",
                )

                st.plotly_chart(
                    chart_layout(fig),
                    use_container_width=True,
                )

            category_available = "category" in filtered.columns

            if category_available:

                category_sales = (
                    filtered.groupby("category", as_index=False)
                    .agg(
                        units_sold=("units_sold", "sum"),
                        revenue=("revenue", "sum"),
                    )
                    .sort_values("revenue", ascending=False)
                )

                col1, col2 = st.columns(2)

                with col1:
                    fig = px.bar(
                        category_sales,
                        x="category",
                        y="revenue",
                        title="Revenue by Category",
                        color="category",
                    )

                    st.plotly_chart(
                        chart_layout(fig),
                        use_container_width=True,
                    )

                with col2:
                    fig = px.pie(
                        category_sales,
                        names="category",
                        values="units_sold",
                        title="Unit Sales Share by Category",
                        hole=0.4,
                    )

                    st.plotly_chart(
                        chart_layout(fig),
                        use_container_width=True,
                    )

            st.subheader("Product Performance")

            product_summary = (
                filtered.groupby(
                    ["product_id", "product_name"],
                    as_index=False,
                )
                .agg(
                    units_sold=("units_sold", "sum"),
                    revenue=("revenue", "sum"),
                )
                .sort_values("units_sold", ascending=False)
            )

            st.dataframe(
                product_summary,
                use_container_width=True,
                hide_index=True,
            )

            safe_download(
                filtered,
                "filtered_sales.csv",
                "Download Filtered Sales",
            )


# ============================================================
# FORECASTING PAGE
# ============================================================

elif page == "Forecasting":

    st.subheader("Sales Forecasting")

    if forecasts is None:
        show_missing_file_message(
            "outputs/forecasts/sales_forecast.csv",
            "python -m src.forecasting",
        )

    else:
        forecasts = prepare_date_column(forecasts)

        required = {
            "date",
            "store_id",
            "product_id",
            "product_name",
            "predicted_units",
        }

        if not required.issubset(forecasts.columns):
            st.error(
                "The forecast file is missing required columns."
            )
            st.stop()

        forecasts["predicted_units"] = pd.to_numeric(
            forecasts["predicted_units"],
            errors="coerce",
        ).fillna(0)

        st.sidebar.subheader("Forecast Filters")

        store_options = sorted(
            forecasts["store_id"].unique().tolist()
        )

        selected_store = st.sidebar.selectbox(
            "Store",
            ["All Stores"] + store_options,
        )

        product_options = sorted(
            forecasts["product_id"].unique().tolist()
        )

        selected_product = st.sidebar.selectbox(
            "Product",
            ["All Products"] + product_options,
        )

        filtered = forecasts.copy()

        if selected_store != "All Stores":
            filtered = filtered[
                filtered["store_id"] == selected_store
            ]

        if selected_product != "All Products":
            filtered = filtered[
                filtered["product_id"] == selected_product
            ]

        if filtered.empty:
            st.info("No forecasts match the selected filters.")

        else:
            total_forecast = filtered["predicted_units"].sum()

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Forecast Records",
                f"{len(filtered):,}",
            )

            col2.metric(
                "Forecast Units",
                f"{total_forecast:,.0f}",
            )

            col3.metric(
                "Forecast Days",
                filtered["date"].nunique(),
            )

            daily_forecast = (
                filtered.groupby("date", as_index=False)
                .agg(
                    predicted_units=("predicted_units", "sum")
                )
            )

            fig = px.line(
                daily_forecast,
                x="date",
                y="predicted_units",
                markers=True,
                title="Forecast Demand Over Time",
                labels={
                    "date": "Forecast Date",
                    "predicted_units": "Predicted Units",
                },
            )

            fig.update_traces(
                line=dict(color="#2475c7", width=3)
            )

            st.plotly_chart(
                chart_layout(fig, 430),
                use_container_width=True,
            )

            st.subheader("Forecast Table")

            display_columns = [
                column
                for column in [
                    "date",
                    "store_id",
                    "product_id",
                    "product_name",
                    "predicted_units",
                    "planning_units",
                ]
                if column in filtered.columns
            ]

            st.dataframe(
                filtered[display_columns],
                use_container_width=True,
                hide_index=True,
            )

            safe_download(
                filtered,
                "sales_forecast.csv",
                "Download Forecast Data",
            )

        st.divider()

        st.subheader("Forecast Model Evaluation")

        if metrics is None:
            show_missing_file_message(
                "outputs/metrics/forecast_metrics.csv",
                "python -m src.forecasting",
            )

        else:
            st.dataframe(
                metrics,
                use_container_width=True,
                hide_index=True,
            )

            if {"model", "MAE", "RMSE"}.issubset(metrics.columns):

                metric_long = metrics.melt(
                    id_vars=["model"],
                    value_vars=["MAE", "RMSE"],
                    var_name="Metric",
                    value_name="Error",
                )

                fig = px.bar(
                    metric_long,
                    x="model",
                    y="Error",
                    color="Metric",
                    barmode="group",
                    title="Forecast Model Error Comparison",
                )

                st.plotly_chart(
                    chart_layout(fig),
                    use_container_width=True,
                )

            safe_download(
                metrics,
                "forecast_metrics.csv",
                "Download Model Metrics",
            )


# ============================================================
# INVENTORY OPTIMIZATION PAGE
# ============================================================

elif page == "Inventory Optimization":

    st.subheader("Inventory Optimization")

    if inventory is None:
        show_missing_file_message(
            "outputs/inventory/inventory_recommendations.csv",
            "python -m src.inventory_optimizer",
        )

    else:
        required = {
            "store_id",
            "product_id",
            "on_hand",
            "reorder_point",
            "target_stock",
            "suggested_order_qty",
            "reorder_required",
        }

        if not required.issubset(inventory.columns):
            st.error(
                "The inventory recommendations file is "
                "missing required columns."
            )
            st.stop()

        for column in [
            "on_hand",
            "reorder_point",
            "target_stock",
            "suggested_order_qty",
        ]:
            inventory[column] = pd.to_numeric(
                inventory[column],
                errors="coerce",
            ).fillna(0)

        inventory["reorder_required"] = (
            inventory["reorder_required"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
        )

        total_order_units = int(
            inventory["suggested_order_qty"].sum()
        )

        reorder_count = int(
            inventory["reorder_required"].sum()
        )

        total_order_cost = (
            float(inventory["estimated_order_cost"].sum())
            if "estimated_order_cost" in inventory.columns
            else 0.0
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Products Requiring Reorder",
            reorder_count,
        )

        col2.metric(
            "Suggested Order Units",
            f"{total_order_units:,}",
        )

        col3.metric(
            "Estimated Order Cost",
            money(total_order_cost),
        )

        st.divider()

        st.subheader("Inventory Level vs Reorder Point")

        chart_data = inventory.copy()

        chart_data["item"] = (
            chart_data["store_id"].astype(str)
            + " / "
            + chart_data["product_id"].astype(str)
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=chart_data["item"],
                y=chart_data["on_hand"],
                name="Current Stock",
                marker_color="#2475c7",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=chart_data["item"],
                y=chart_data["reorder_point"],
                name="Reorder Point",
                mode="lines+markers",
                line=dict(
                    color="#e28b26",
                    width=2,
                ),
            )
        )

        fig.update_layout(
            title="Current Stock Compared with Reorder Point",
            xaxis_title="Store / Product",
            yaxis_title="Units",
            xaxis_tickangle=-60,
            barmode="group",
        )

        st.plotly_chart(
            chart_layout(fig, 500),
            use_container_width=True,
        )

        st.subheader("Products Requiring Reorder")

        reorder_items = inventory[
            inventory["reorder_required"]
        ].copy()

        if reorder_items.empty:
            st.success(
                "No products are currently flagged for reorder."
            )

        else:
            st.dataframe(
                reorder_items,
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("All Inventory Recommendations")

        st.dataframe(
            inventory,
            use_container_width=True,
            hide_index=True,
        )

        safe_download(
            inventory,
            "inventory_recommendations.csv",
            "Download Inventory Recommendations",
        )


# ============================================================
# INVENTORY SIMULATION PAGE
# ============================================================

elif page == "Inventory Simulation":

    st.subheader("Inventory Simulation")

    if simulation_daily is None:
        show_missing_file_message(
            "outputs/inventory/daily_inventory_simulation.csv",
            "python -m src.inventory_simulator",
        )

    else:
        simulation_daily = prepare_date_column(
            simulation_daily
        )

        required = {
            "date",
            "store_id",
            "product_id",
            "customer_demand",
            "fulfilled_sales",
            "lost_sales",
            "closing_stock",
            "order_qty",
            "delivery_qty",
        }

        if not required.issubset(simulation_daily.columns):
            st.error(
                "The daily simulation file is missing "
                "required columns."
            )
            st.stop()

        numeric_columns = [
            "customer_demand",
            "fulfilled_sales",
            "lost_sales",
            "closing_stock",
            "order_qty",
            "delivery_qty",
        ]

        for column in numeric_columns:
            simulation_daily[column] = pd.to_numeric(
                simulation_daily[column],
                errors="coerce",
            ).fillna(0)

        total_demand = int(
            simulation_daily["customer_demand"].sum()
        )

        total_fulfilled = int(
            simulation_daily["fulfilled_sales"].sum()
        )

        total_lost = int(
            simulation_daily["lost_sales"].sum()
        )

        fulfilment_rate = (
            total_fulfilled / total_demand * 100
            if total_demand > 0
            else 100.0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Customer Demand",
            f"{total_demand:,}",
        )

        col2.metric(
            "Fulfilled Sales",
            f"{total_fulfilled:,}",
        )

        col3.metric(
            "Lost Sales",
            f"{total_lost:,}",
        )

        col4.metric(
            "Fulfilment Rate",
            f"{fulfilment_rate:.2f}%",
        )

        st.divider()

        st.subheader("Daily Demand and Fulfilled Sales")

        daily_summary = (
            simulation_daily.groupby("date", as_index=False)
            .agg(
                customer_demand=("customer_demand", "sum"),
                fulfilled_sales=("fulfilled_sales", "sum"),
                lost_sales=("lost_sales", "sum"),
            )
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=daily_summary["date"],
                y=daily_summary["customer_demand"],
                mode="lines+markers",
                name="Customer Demand",
                line=dict(color="#2475c7", width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=daily_summary["date"],
                y=daily_summary["fulfilled_sales"],
                mode="lines+markers",
                name="Fulfilled Sales",
                line=dict(color="#168f80", width=2),
            )
        )

        fig.add_trace(
            go.Bar(
                x=daily_summary["date"],
                y=daily_summary["lost_sales"],
                name="Lost Sales",
                marker_color="#e28b26",
                opacity=0.65,
            )
        )

        fig.update_layout(
            title="Daily Simulated Inventory Performance",
            xaxis_title="Date",
            yaxis_title="Units",
        )

        st.plotly_chart(
            chart_layout(fig, 430),
            use_container_width=True,
        )

        st.subheader("Closing Stock Over Time")

        stock_summary = (
            simulation_daily.groupby(
                ["date", "store_id"],
                as_index=False,
            )
            .agg(
                closing_stock=("closing_stock", "sum")
            )
        )

        fig = px.line(
            stock_summary,
            x="date",
            y="closing_stock",
            color="store_id",
            markers=True,
            title="Combined Closing Stock by Store",
        )

        st.plotly_chart(
            chart_layout(fig, 430),
            use_container_width=True,
        )

        st.subheader("Simulation Summary by Product")

        if simulation_summary is not None:

            st.dataframe(
                simulation_summary,
                use_container_width=True,
                hide_index=True,
            )

        safe_download(
            simulation_daily,
            "daily_inventory_simulation.csv",
            "Download Daily Simulation",
        )

        if simulation_summary is not None:
            safe_download(
                simulation_summary,
                "inventory_simulation_summary.csv",
                "Download Simulation Summary",
            )


# ============================================================
# DATA EXPLORER PAGE
# ============================================================

elif page == "Data Explorer":

    st.subheader("Project Data Explorer")

    dataset_options = {
        "Cleaned Sales": sales,
        "Future Forecasts": forecasts,
        "Forecast Metrics": metrics,
        "Inventory Recommendations": inventory,
        "Daily Inventory Simulation": simulation_daily,
        "Inventory Simulation Summary": simulation_summary,
    }

    selected_dataset = st.selectbox(
        "Choose a dataset",
        list(dataset_options.keys()),
    )

    selected_data = dataset_options[selected_dataset]

    if selected_data is None:

        st.warning(
            f"The dataset '{selected_dataset}' is not available. "
            "Generate its output using the relevant project module."
        )

    else:

        st.write(
            f"**Rows:** {selected_data.shape[0]:,} "
            f"| **Columns:** {selected_data.shape[1]:,}"
        )

        st.dataframe(
            selected_data,
            use_container_width=True,
            hide_index=True,
        )

        safe_download(
            selected_data,
            selected_dataset.lower().replace(" ", "_") + ".csv",
            "Download Selected Dataset",
        )

        st.subheader("Dataset Information")

        info = pd.DataFrame(
            {
                "Column": selected_data.columns,
                "Data Type": [
                    str(dtype)
                    for dtype in selected_data.dtypes
                ],
                "Missing Values": [
                    int(selected_data[column].isna().sum())
                    for column in selected_data.columns
                ],
                "Unique Values": [
                    int(selected_data[column].nunique())
                    for column in selected_data.columns
                ],
            }
        )

        st.dataframe(
            info,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Retail Sales Forecasting & Inventory Optimization System | "
    "Built with Python, Pandas, Scikit-learn, Plotly, and Streamlit | "
    "Synthetic educational dataset"
)
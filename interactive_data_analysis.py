import marimo

# Author: 23f3001400@ds.study.iitm.ac.in  <-- verification email as a comment
# This Marimo app demonstrates:
# - At least two cells with variable dependencies
# - An interactive slider widget
# - Dynamic markdown output based on widget state
# - Explicit comments documenting the data flow between cells
# Data Flow Overview:
#   Cell A (data generation) -> Cell C (filtering & features) -> Cell D/E/F (analysis, markdown, viz)
#   Cell B (widgets) --------^

__generated_with = "0.8.0"
app = marimo.App(width="medium")


@app.cell
def __():
    """
    Cell A: Imports & intro markdown (root deps)
    - Provides shared imports for downstream cells
    - Renders static intro markdown
    Data Flow: defines (mo, np, pd, px, go, datetime, timedelta) -> used by cells B–F
    """
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import datetime, timedelta

    mo.md(
        f"""
        # Interactive Data Analysis with Marimo

        **Author:** 23f3001400@ds.study.iitm.ac.in  
        **Date:** August 16, 2025

        This app showcases Marimo's reactive programming with:
        - Variable dependencies between cells
        - Interactive widgets
        - Dynamic markdown output
        - Real-time data visualization
        """
    )
    return datetime, go, mo, np, pd, px, timedelta


@app.cell
def __(np, pd):
    """
    Cell C0: Base data generation
    - Creates reproducible daily sales data with trend + seasonality + noise
    Data Flow: outputs raw_data -> consumed by Cell C (filtering)
    """
    np.random.seed(42)

    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    base_sales = np.random.normal(1000, 200, len(dates))
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    trend = np.linspace(0, 500, len(dates))
    noise = np.random.normal(0, 50, len(dates))

    daily_sales = np.maximum(base_sales * seasonal_factor + trend + noise, 0)

    raw_data = pd.DataFrame({
        "date": dates,
        "daily_sales": daily_sales,
        "month": dates.month,
        "day_of_week": dates.day_name(),
        "quarter": dates.quarter,
    })

    print(f"Generated dataset with {len(raw_data)} records")
    print(f"Sales range: ${raw_data['daily_sales'].min():.2f} - ${raw_data['daily_sales'].max():.2f}")
    raw_data
    return base_sales, daily_sales, dates, noise, raw_data, seasonal_factor, trend


@app.cell
def __(mo):
    """
    Cell B: Interactive controls (widgets)
    - Sliders & dropdown drive downstream filtering, analysis, and output
    Data Flow: outputs sample_size_slider, analysis_type, threshold_slider -> used by Cells C/D/E/F
    """
    sample_size_slider = mo.ui.slider(start=50, stop=365, step=10, value=200, label="Sample Size (days)")
    analysis_type = mo.ui.dropdown(options=["trend", "seasonal", "quarterly"], value="trend", label="Analysis Type")
    threshold_slider = mo.ui.slider(start=500, stop=2000, step=50, value=1200, label="Sales Threshold ($)")

    mo.md(f"""
    ## Interactive Controls

    Use these controls to modify the analysis:

    {sample_size_slider}

    {analysis_type}

    {threshold_slider}
    """)
    return analysis_type, sample_size_slider, threshold_slider


@app.cell
def __(raw_data, sample_size_slider):
    """
    Cell C: Data processing based on widget state
    - Filters raw_data by sample_size_slider
    - Computes rolling averages & categorical labels
    Data Flow: (raw_data + sample_size_slider) -> filtered_data -> used by Cells D/E/F
    """
    sample_size = sample_size_slider.value

    filtered_data = raw_data.head(sample_size).copy()
    filtered_data["rolling_avg_7"] = filtered_data["daily_sales"].rolling(window=7, center=True).mean()
    filtered_data["rolling_avg_30"] = filtered_data["daily_sales"].rolling(window=30, center=True).mean()
    filtered_data["sales_category"] = filtered_data["daily_sales"].apply(
        lambda x: "High" if x > 1200 else ("Medium" if x > 800 else "Low")
    )

    print(f"Filtered data contains {len(filtered_data)} records")
    print(f"Average daily sales: ${filtered_data['daily_sales'].mean():.2f}")

    filtered_data
    return filtered_data, sample_size


@app.cell
def __(analysis_type, filtered_data, threshold_slider):
    """
    Cell D: Analysis logic driven by widgets
    - Branches on analysis_type (trend/seasonal/quarterly)
    - Uses threshold_slider to count days above threshold
    Data Flow: (filtered_data + analysis_type + threshold_slider) -> analysis_results
    """
    import numpy as np

    analysis_choice = analysis_type.value
    threshold_value = threshold_slider.value

    # Predefine branch variables to avoid NameError across returns
    correlation = None
    trend_direction = None
    monthly_avg = None
    quarterly_avg = None
    best_month = None
    worst_month = None
    best_quarter = None

    if analysis_choice == "trend":
        # Correlate sales with time index
        idx = np.arange(len(filtered_data))
        correlation = float(np.corrcoef(idx, filtered_data["daily_sales"].to_numpy())[0, 1])
        trend_direction = "increasing" if correlation > 0 else "decreasing"
        analysis_results = {
            "type": "Trend Analysis",
            "correlation": correlation,
            "direction": trend_direction,
            "avg_sales": float(filtered_data["daily_sales"].mean()),
            "above_threshold": int((filtered_data["daily_sales"] > threshold_value).sum()),
        }

    elif analysis_choice == "seasonal":
        monthly_avg = filtered_data.groupby("month")["daily_sales"].mean()
        best_month = int(monthly_avg.idxmax())
        worst_month = int(monthly_avg.idxmin())
        analysis_results = {
            "type": "Seasonal Analysis",
            "best_month": best_month,
            "worst_month": worst_month,
            "seasonal_variation": float(monthly_avg.std()),
            "above_threshold": int((filtered_data["daily_sales"] > threshold_value).sum()),
        }

    else:  # quarterly
        quarterly_avg = filtered_data.groupby("quarter")["daily_sales"].mean()
        best_quarter = int(quarterly_avg.idxmax())
        analysis_results = {
            "type": "Quarterly Analysis",
            "best_quarter": f"Q{best_quarter}",
            "quarterly_growth": float(quarterly_avg.pct_change().mean() * 100),
            "total_quarters": int(len(quarterly_avg)),
            "above_threshold": int((filtered_data["daily_sales"] > threshold_value).sum()),
        }

    analysis_results
    return (
        analysis_choice,
        analysis_results,
        best_month,
        best_quarter,
        correlation,
        monthly_avg,
        quarterly_avg,
        threshold_value,
        trend_direction,
        worst_month,
    )


@app.cell
def __(analysis_results, mo, sample_size, threshold_value):
    """
    Cell E: Dynamic markdown driven by widget state
    - Renders different summaries based on analysis_results['type']
    Data Flow: (analysis_results + sample_size + threshold_value) -> markdown output
    """
    results = analysis_results

    if results["type"] == "Trend Analysis":
        dynamic_content = f"""
        ## 📈 {results['type']} Results

        **Sample Size:** {sample_size} days  
        **Threshold:** ${threshold_value}

        ### Key Findings:
        - **Trend Direction:** Sales are {results['direction']} over time
        - **Correlation Coefficient:** {results['correlation']:.4f}
        - **Average Daily Sales:** ${results['avg_sales']:.2f}
        - **Days Above Threshold:** {results['above_threshold']} days

        {"📈 **Positive trend detected!**" if results['correlation'] > 0 else "📉 **Negative trend detected**"}
        """

    elif results["type"] == "Seasonal Analysis":
        dynamic_content = f"""
        ## 🌟 {results['type']} Results

        **Sample Size:** {sample_size} days  
        **Threshold:** ${threshold_value}

        ### Key Findings:
        - **Best Performing Month:** Month {results['best_month']}
        - **Worst Performing Month:** Month {results['worst_month']}
        - **Seasonal Variation (Std Dev):** ${results['seasonal_variation']:.2f}
        - **Days Above Threshold:** {results['above_threshold']} days

        🎯 **Seasonal patterns identified in the data!**
        """

    else:  # Quarterly Analysis
        dynamic_content = f"""
        ## 📊 {results['type']} Results

        **Sample Size:** {sample_size} days  
        **Threshold:** ${threshold_value}

        ### Key Findings:
        - **Best Quarter:** {results['best_quarter']}
        - **Average Quarterly Growth:** {results['quarterly_growth']:.2f}%
        - **Quarters Analyzed:** {results['total_quarters']}
        - **Days Above Threshold:** {results['above_threshold']} days

        {"🚀 **Positive growth trajectory!**" if results['quarterly_growth'] > 0 else "⚠️ **Declining quarterly performance**"}
        """

    mo.md(dynamic_content)
    return dynamic_content, results


@app.cell
def __(filtered_data, mo, px, threshold_value):
    """
    Cell F: Interactive visualization
    - Plots daily sales, rolling averages, and a threshold marker
    Data Flow: (filtered_data + threshold_value) -> plotly figure output
    """
    fig = px.line(
        filtered_data,
        x="date",
        y="daily_sales",
        title="Daily Sales Over Time with Interactive Threshold",
        labels={"daily_sales": "Daily Sales ($)", "date": "Date"},
        line_shape="linear",
    )

    # Optional overlays if present
    if "rolling_avg_7" in filtered_data.columns:
        fig.add_scatter(
            x=filtered_data["date"],
            y=filtered_data["rolling_avg_7"],
            mode="lines",
            name="7-day Average",
            line=dict(dash="dash"),
        )

    if "rolling_avg_30" in filtered_data.columns:
        fig.add_scatter(
            x=filtered_data["date"],
            y=filtered_data["rolling_avg_30"],
            mode="lines",
            name="30-day Average",
            line=dict(dash="dot"),
        )

    # Threshold line
    fig.add_hline(y=threshold_value, line_dash="dash", annotation_text=f"Threshold: ${threshold_value}")

    # Highlight points above threshold
    above = filtered_data[filtered_data["daily_sales"] > threshold_value]
    if len(above) > 0:
        fig.add_scatter(
            x=above["date"],
            y=above["daily_sales"],
            mode="markers",
            name=f"Above ${threshold_value}",
        )

    mo.ui.plotly(fig)
    return above, fig


@app.cell
def __(mo):
    """
    Cell Z: Footer / documentation
    - Summarizes dependency graph and usage
    """
    mo.md(
        """
        ---

        ## 📋 Notebook Structure & Dependencies

        1. **Cell A** (Imports & intro) → provides shared imports to **B–F**
        2. **Cell C0** (Base data) → feeds **Cell C** (filtering & features)
        3. **Cell B** (Widgets) → feeds **Cells C, D, E, F**
        4. **Cell C** (Filtered Data) → feeds **Cells D, E, F**
        5. **Cell D** (Analysis) → feeds **Cell E** (dynamic markdown)

        ### 🔄 Reactive Features
        - Changing any widget re-evaluates downstream cells
        - Markdown & charts update in real time based on widget state

        **Author:** 23f3001400@ds.study.iitm.ac.in  
        **Created with:** Marimo v0.8.0  
        **Date:** August 16, 2025
        """
    )
    return


if __name__ == "__main__":
    app.run()

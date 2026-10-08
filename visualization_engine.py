import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def plot_bar_chart(df, x_col, y_col, title=None):
    """Bar chart — e.g. Revenue by Region"""
    fig = px.bar(
        df, x=x_col, y=y_col,
        title=title or f"{y_col} by {x_col}",
        color=x_col,
        template="plotly_dark"
    )
    return fig


def plot_line_chart(trend_data, date_col, value_col, title=None):
    """Line chart — e.g. Revenue over Time"""
    df = pd.DataFrame(trend_data)
    fig = px.line(
        df, x=date_col, y=value_col,
        title=title or f"{value_col} Over Time",
        markers=True,
        template="plotly_dark"
    )
    return fig


def plot_pie_chart(df, names_col, values_col, title=None):
    """Pie chart — e.g. Product Sales Distribution"""
    fig = px.pie(
        df, names=names_col, values=values_col,
        title=title or f"{names_col} Distribution",
        template="plotly_dark"
    )
    return fig


def plot_histogram(df, col, title=None):
    """Histogram — e.g. Revenue Distribution"""
    fig = px.histogram(
        df, x=col,
        title=title or f"Distribution of {col}",
        nbins=30,
        template="plotly_dark"
    )
    return fig


def plot_scatter(df, x_col, y_col, color_col=None, title=None):
    """Scatter plot — e.g. Quantity vs Revenue"""
    fig = px.scatter(
        df, x=x_col, y=y_col,
        color=color_col,
        title=title or f"{x_col} vs {y_col}",
        template="plotly_dark"
    )
    return fig


def generate_auto_charts(df, column_types, trends):
    """
    Master function — auto-decides which charts to generate
    based on available column types. Returns a list of figures.
    """
    charts = []
    numeric_cols = column_types["numeric"]
    categorical_cols = column_types["categorical"]
    date_cols = column_types["date"]

    # Bar chart: first categorical vs first numeric
    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        grouped = df.groupby(cat_col)[num_col].sum().reset_index()
        charts.append(plot_bar_chart(grouped, cat_col, num_col))

    # Pie chart: first categorical column distribution
    if categorical_cols:
        cat_col = categorical_cols[0]
        counts = df[cat_col].value_counts().reset_index()
        counts.columns = [cat_col, "count"]
        charts.append(plot_pie_chart(counts, cat_col, "count"))

    # Histogram: all numeric columns
    for col in numeric_cols[:2]:  # limit to first 2 to avoid overload
        charts.append(plot_histogram(df, col))

    # Line chart: if trend data exists
    if trends and date_cols:
        date_col = date_cols[0]
        for num_col in numeric_cols[:1]:  # first numeric only
            if num_col in trends:
                charts.append(plot_line_chart(trends[num_col], date_col, num_col))

    # Scatter plot: first two numeric columns
    if len(numeric_cols) >= 2:
        charts.append(plot_scatter(df, numeric_cols[0], numeric_cols[1]))

    return charts

import pandas as pd
import numpy as np


def get_dataset_summary(df):
    """
    Returns basic structural info about the dataset.
    """
    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": df.isnull().sum().sum()
    }
    return summary


def get_descriptive_statistics(df, numeric_cols):
    """
    Returns mean, median, min, max, std for all numeric columns.
    """
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "mean": round(df[col].mean(), 2),
            "median": round(df[col].median(), 2),
            "min": round(df[col].min(), 2),
            "max": round(df[col].max(), 2),
            "std_dev": round(df[col].std(), 2)
        }
    return stats


def get_distribution_analysis(df, categorical_cols):
    """
    Returns value counts + percentage distribution for categorical columns.
    """
    distribution = {}
    for col in categorical_cols:
        counts = df[col].value_counts()
        percentages = round(df[col].value_counts(normalize=True) * 100, 2)
        distribution[col] = {
            "counts": counts.to_dict(),
            "percentages": percentages.to_dict()
        }
    return distribution


def get_trend_analysis(df, date_cols, numeric_cols):
    """
    If date columns exist, analyze trends of numeric columns over time.
    """
    trends = {}

    if not date_cols or not numeric_cols:
        return trends

    date_col = date_cols[0]  # Use the first detected date column

    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df_sorted = df.sort_values(date_col)

        for num_col in numeric_cols:
            monthly = (
                df_sorted.groupby(df_sorted[date_col].dt.to_period("M"))[num_col]
                .sum()
                .reset_index()
            )
            monthly[date_col] = monthly[date_col].astype(str)
            trends[num_col] = monthly.to_dict(orient="records")

    except Exception as e:
        trends["error"] = str(e)

    return trends


def run_full_analysis(df, column_types):
    """
    Master function — runs all analysis and returns a single result dict.
    """
    numeric_cols = column_types["numeric"]
    categorical_cols = column_types["categorical"]
    date_cols = column_types["date"]

    return {
        "summary": get_dataset_summary(df),
        "statistics": get_descriptive_statistics(df, numeric_cols),
        "distribution": get_distribution_analysis(df, categorical_cols),
        "trends": get_trend_analysis(df, date_cols, numeric_cols)
    }

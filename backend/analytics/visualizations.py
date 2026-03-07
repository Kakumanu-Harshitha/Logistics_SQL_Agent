"""
analytics/visualizations.py
Plotly chart generation with auto-selection of chart type.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional


def auto_visualize(df: pd.DataFrame, question: str = "") -> Optional[go.Figure]:
    """
    Auto-select and generate the most appropriate Plotly chart
    for a given DataFrame, based on column types and shape.
    """
    if df is None or df.empty or len(df.columns) < 1:
        return None

    cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    q_lower = question.lower()

    # --- Time series ---
    if date_cols and num_cols:
        return _time_series(df, date_cols[0], num_cols[0], question)

    # --- Heatmap for delay/matrix ---
    if "heatmap" in q_lower or (len(cat_cols) >= 2 and num_cols):
        try:
            return _heatmap(df, cat_cols, num_cols[0], question)
        except Exception:
            pass

    # --- Bar chart (1 categorical + 1 numeric) ---
    if cat_cols and num_cols:
        return _bar_chart(df, cat_cols[0], num_cols[0], question)

    # --- Scatter (2 numeric columns) ---
    if len(num_cols) >= 2:
        return _scatter(df, num_cols[0], num_cols[1], cat_cols[0] if cat_cols else None, question)

    # --- Single numeric: histogram ---
    if num_cols:
        return _histogram(df, num_cols[0], question)

    return None


def _bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    df_plot = df.nlargest(min(20, len(df)), y_col) if len(df) > 20 else df
    color_scale = px.colors.sequential.Blues_r

    fig = px.bar(
        df_plot,
        x=x_col,
        y=y_col,
        title=title or f"{y_col} by {x_col}",
        text=y_col,
        color=y_col,
        color_continuous_scale="Blues",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="#ffffff", family="Inter"),
        title_font=dict(size=18, color="#60a5fa"),
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="#1e293b", tickangle=-30),
        yaxis=dict(gridcolor="#1e293b"),
        margin=dict(t=60, b=80, l=60, r=20),
    )
    return fig


def _time_series(df: pd.DataFrame, date_col: str, y_col: str, title: str) -> go.Figure:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[y_col],
        mode="lines+markers",
        line=dict(color="#60a5fa", width=2),
        marker=dict(size=4, color="#93c5fd"),
        name=y_col,
    ))
    fig.update_layout(
        title=title or f"{y_col} over time",
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="#ffffff", family="Inter"),
        title_font=dict(size=18, color="#60a5fa"),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        margin=dict(t=60, b=50, l=60, r=20),
    )
    return fig


def _heatmap(df: pd.DataFrame, cat_cols: list, val_col: str, title: str) -> go.Figure:
    pivot = df.pivot_table(index=cat_cols[0], columns=cat_cols[1], values=val_col, aggfunc="mean")
    fig = px.imshow(
        pivot,
        title=title or "Heatmap",
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig.update_layout(
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="#ffffff", family="Inter"),
        title_font=dict(size=18, color="#60a5fa"),
        margin=dict(t=60, b=50, l=80, r=20),
    )
    return fig


def _scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: Optional[str], title: str) -> go.Figure:
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title or f"{y_col} vs {x_col}",
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig.update_layout(
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="#ffffff", family="Inter"),
        title_font=dict(size=18, color="#60a5fa"),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        margin=dict(t=60, b=50, l=60, r=20),
    )
    return fig


def _histogram(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    fig = px.histogram(
        df,
        x=col,
        title=title or f"Distribution of {col}",
        nbins=30,
        color_discrete_sequence=["#60a5fa"],
    )
    fig.update_layout(
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="#ffffff", family="Inter"),
        title_font=dict(size=18, color="#60a5fa"),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        bargap=0.05,
        margin=dict(t=60, b=50, l=60, r=20),
    )
    return fig


def delivery_delay_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
    """Specialized heatmap for delivery delay rates by region and shipping mode."""
    if df is None or df.empty:
        return None
    if "late_delivery_risk" not in df.columns:
        return auto_visualize(df)
    try:
        pivot = df.pivot_table(
            index="order_region" if "order_region" in df.columns else df.columns[0],
            columns="shipping_mode" if "shipping_mode" in df.columns else df.columns[1],
            values="late_delivery_risk",
            aggfunc="mean",
        )
        fig = px.imshow(
            pivot * 100,
            title="Delivery Delay Rate (%) by Region & Shipping Mode",
            color_continuous_scale="RdYlGn_r",
            aspect="auto",
            labels=dict(color="Delay %"),
        )
        fig.update_layout(
            plot_bgcolor="#0f1117",
            paper_bgcolor="#0f1117",
            font=dict(color="#ffffff", family="Inter"),
            title_font=dict(size=18, color="#f87171"),
            margin=dict(t=60, b=80, l=120, r=20),
        )
        return fig
    except Exception:
        return auto_visualize(df)


def driver_performance_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """Ranked horizontal bar chart for driver performance."""
    if df is None or df.empty:
        return None
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not num_cols or not cat_cols:
        return auto_visualize(df)
    df_sorted = df.nlargest(15, num_cols[0])
    fig = px.bar(
        df_sorted,
        y=cat_cols[0],
        x=num_cols[0],
        orientation="h",
        title="Driver Performance Ranking",
        color=num_cols[0],
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="#ffffff", family="Inter"),
        title_font=dict(size=18, color="#60a5fa"),
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed", gridcolor="#1e293b"),
        xaxis=dict(gridcolor="#1e293b"),
        margin=dict(t=60, b=50, l=150, r=20),
    )
    return fig

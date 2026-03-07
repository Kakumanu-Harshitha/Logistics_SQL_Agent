"""
analytics/insights.py
Pandas-based business insight generation from query result DataFrames.
"""
import pandas as pd
import numpy as np
from typing import Optional
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def generate_insight(df: pd.DataFrame, question: str) -> str:
    """
    Generate a natural language business insight from a DataFrame result
    using Gemini to interpret the data.
    """
    if df is None or df.empty:
        return "No data returned for this query."

    # Build a compact representation
    preview = df.head(10).to_string(index=False)
    shape_info = f"Result: {len(df)} rows × {len(df.columns)} columns"
    cols = ", ".join(df.columns.tolist())

    # Statistical summary for numeric columns
    stats_lines = []
    for col in df.select_dtypes(include=[np.number]).columns:
        try:
            stats_lines.append(
                f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}, "
                f"mean={df[col].mean():.2f}"
            )
        except Exception:
            pass
    stats = "\n".join(stats_lines) if stats_lines else "No numeric statistics."

    prompt = f"""You are a logistics business analyst. Based on the SQL query result below,
generate a concise business insight (3-5 sentences) highlighting key findings,
patterns, anomalies, or actionable recommendations.

Original question: {question}

{shape_info}
Columns: {cols}

Sample data (up to 10 rows):
{preview}

Numeric statistics:
{stats}

Write a professional business insight paragraph. Be specific and quantitative where possible."""

    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=GROQ_API_KEY,
            temperature=0.3,
            max_tokens=512,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        return _fallback_insight(df, question)


def _fallback_insight(df: pd.DataFrame, question: str) -> str:
    """Rule-based fallback insight when LLM is unavailable."""
    if df.empty:
        return "No results found for your query."

    insights = [f"Query returned {len(df)} records."]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if num_cols:
        primary = num_cols[0]
        max_val = df[primary].max()
        min_val = df[primary].min()
        insights.append(f"The highest value of {primary} is {max_val:.2f} and the lowest is {min_val:.2f}.")

        # Top performer
        if len(df.columns) >= 2:
            label_col = df.columns[0]
            top_row = df.loc[df[primary].idxmax()]
            insights.append(f"Top performer: {top_row[label_col]} with {primary} = {top_row[primary]:.2f}.")

    return " ".join(insights)


def compute_route_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute route efficiency score if DataFrame has the right columns.
    efficiency_score = successful_deliveries / total_delivery_days
    """
    required = {"delivery_status", "days_for_shipping_real"}
    if not required.issubset(set(df.columns)):
        return df

    df = df.copy()
    success_mask = df["delivery_status"].str.lower().str.contains("on time|advance", na=False)
    df["is_successful"] = success_mask.astype(int)
    if "days_for_shipping_real" in df.columns:
        df["efficiency_score"] = df["is_successful"] / df["days_for_shipping_real"].replace(0, np.nan)
    return df

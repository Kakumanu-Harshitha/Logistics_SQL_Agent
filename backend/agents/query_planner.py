"""
agents/query_planner.py
Multi-step query decomposition for complex analytical questions.
"""
import os
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

PLANNER_PROMPT = """You are an expert data analyst specializing in logistics and supply chain analytics.

Your task is to determine if a user's question requires MULTIPLE SQL queries to answer,
or if it can be answered with a single SQL query.

If it requires multiple steps, break it down into numbered sub-questions.
If a single query suffices, respond with: SINGLE_QUERY

Examples:
Input: "Which city has the highest delivery delay percentage?"
Output:
1. Total deliveries grouped by city
2. Late deliveries (where late_delivery_risk=1) grouped by city
3. Calculate delay percentage = late_deliveries / total_deliveries * 100 per city

Input: "Which drivers have the highest rating?"
Output: SINGLE_QUERY

Always respond with either "SINGLE_QUERY" or a numbered list of sub-questions only.
Do not include any other text.
"""


def _build_planner_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
        temperature=0.0,
        max_tokens=512,
    )


def plan_query(question: str) -> list:
    """
    Determine if a question needs multi-step planning.
    Currently hardcoded to return a single query to enforce single-query execution.
    """
    return [question]


def is_followup_question(question: str, history: list) -> bool:
    """
    Detect if question is a follow-up to a previous one.
    """
    followup_markers = [
        "show me", "also show", "and what about", "what about",
        "now show", "for those", "of those", "for the same",
        "break it down", "more details", "same but", "filter by",
        "only show", "top 5", "top 10", "bottom",
    ]
    q_lower = question.lower()
    has_marker = any(marker in q_lower for marker in followup_markers)
    has_history = len(history) > 0
    return has_marker and has_history

"""
api/routes.py
FastAPI router with all analytics endpoints.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.sql_agent import get_agent
from agents.query_planner import plan_query, is_followup_question
from database.schema_loader import get_schema_string, get_schema_dict
from analytics.insights import generate_insight
from analytics.visualizations import auto_visualize
from ml.delay_prediction import predict_delay, train_model, model_is_trained
from database.db_connection import test_connection

router = APIRouter()


# ─── Request / Response Models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"
    clear_history: Optional[bool] = False


class QueryResponse(BaseModel):
    question: str
    sql: str
    row_count: int
    columns: List[str]
    results: List[Dict[str, Any]]
    chart_json: Optional[str] = None
    insight: str
    error: Optional[str] = None
    steps: Optional[List[str]] = None


class DelayPredictRequest(BaseModel):
    distance_km: float = 200.0
    days_scheduled: int = 3
    traffic_level: str = "Medium"
    shipping_mode: str = "Standard Class"
    experience_years: int = 3
    rating: float = 3.5
    hour_of_day: int = 12
    day_of_week: int = 1


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    db_ok = test_connection()
    return {
        "status": "ok",
        "database": "connected" if db_ok else "disconnected",
        "model_trained": model_is_trained(),
    }


@router.get("/schema")
async def get_schema():
    try:
        schema_dict = get_schema_dict()
        schema_str = get_schema_string()
        return {"schema_dict": schema_dict, "schema_text": schema_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sample-questions")
async def sample_questions():
    return {
        "questions": [
            "Which delivery routes have the highest delay rate?",
            "Which drivers complete deliveries the fastest?",
            "Which warehouse processes the most orders?",
            "What is the average delivery time per city?",
            "Which shipping mode has the worst late delivery rate?",
            "Show me the top 10 customers by order value",
            "What is the total sales by product category?",
            "Which regions have the most canceled orders?",
            "Compare on-time vs late deliveries by region",
            "What is the monthly order trend over the past year?",
        ]
    }


@router.post("/query", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    agent = get_agent()

    if req.clear_history:
        agent.clear_history()

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Multi-step planning
    steps = plan_query(question)
    is_multi = len(steps) > 1

    logger.info(f"Processing query: {question} (session_id: {req.session_id})")
    try:
        if is_multi:
            # Execute each sub-step and combine results
            combined_dfs = []
            sqls = []
            for step_q in steps:
                sql, df, err = await agent.run(step_q, session_id=req.session_id)
                sqls.append(sql)
                if not df.empty:
                    combined_dfs.append(df)

            final_sql = "\n-- NEXT STEP --\n".join(sqls)
            final_df = combined_dfs[-1] if combined_dfs else pd.DataFrame()
        else:
            final_sql, final_df, err = await agent.run(question, session_id=req.session_id)
            if err:
                logger.error(f"Agent failed to analyze query: {err}")
                return QueryResponse(
                    question=question,
                    sql=final_sql,
                    row_count=0,
                    columns=[],
                    results=[],
                    insight=f"Query failed: {err}",
                    error=err,
                    steps=steps,
                )

        # Sanitize DataFrame for JSON serialization
        if not final_df.empty:
            final_df = final_df.where(pd.notnull(final_df), None)
            for col in final_df.select_dtypes(include=["datetime64", "datetimetz"]).columns:
                final_df[col] = final_df[col].astype(str)

        results = final_df.to_dict(orient="records") if not final_df.empty else []
        columns = list(final_df.columns) if not final_df.empty else []

        # Visualization
        chart_json = None
        if not final_df.empty:
            fig = auto_visualize(final_df, question)
            if fig is not None:
                chart_json = fig.to_json()

        # Insight generation
        insight = generate_insight(final_df, question)

        # Store in permanent memory
        if req.session_id:
            from database.db_connection import get_async_engine
            from sqlalchemy import text
            engine = get_async_engine()
            async with engine.begin() as conn:
                await conn.execute(text(
                    "INSERT INTO chats (session_id, user_question, generated_sql) VALUES (:session_id, :q, :sql)"
                ), {"session_id": req.session_id, "q": question, "sql": final_sql})

        return QueryResponse(
            question=question,
            sql=final_sql,
            row_count=len(final_df),
            columns=columns,
            results=results,
            chart_json=chart_json,
            insight=insight,
            error=None,
            steps=steps,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-delay")
async def predict_delay_endpoint(req: DelayPredictRequest):
    if not model_is_trained():
        raise HTTPException(
            status_code=400,
            detail="Model not trained. POST to /train-model first."
        )
    result = predict_delay(req.dict())
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.post("/train-model")
async def train_model_endpoint():
    try:
        result = train_model()
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-history")
async def clear_conversation_history():
    agent = get_agent()
    agent.clear_history()
    return {"message": "Conversation history cleared."}

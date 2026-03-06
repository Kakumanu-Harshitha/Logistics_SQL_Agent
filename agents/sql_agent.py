"""
agents/sql_agent.py
LangChain + Gemini SQL generation agent with self-correction and conversation memory.
"""
import os
from typing import Optional, Tuple
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import text

from database.db_connection import get_engine
from database.schema_loader import get_semantic_schema_string, get_table_names
from utils.query_validator import validate_sql, clean_sql
import logging

logger = logging.getLogger(__name__)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MAX_RETRY = 3

# Priority list for models (High tier -> Small tier)
FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
]


def _build_llm(model_name: str = "llama-3.3-70b-versatile") -> ChatGroq:
    return ChatGroq(
        model=model_name,
        groq_api_key=GROQ_API_KEY,
        temperature=0.1,
        max_tokens=2048,
    )


SYSTEM_PROMPT = """You are a senior data engineer designing optimized PostgreSQL queries for a logistics analytics system.

The system converts natural language questions into SQL queries executed on a PostgreSQL database containing supply chain data (orders, deliveries, routes, products, customers, shipping).

Your task is to generate ONE optimized SQL query that answers the user question.

Database Schema:
{schema}

REQUIREMENTS
1. Generate only ONE SQL query per question. Do NOT generate intermediate exploratory queries.
2. Combine all required metrics in a single query using SQL aggregation.
3. Use efficient PostgreSQL constructs such as: COUNT(*) FILTER (WHERE condition), CASE WHEN, GROUP BY aggregation.
4. Avoid redundant scans of the same tables.
5. Ensure queries work efficiently on datasets larger than 100k rows.

COMPLEX QUERY HANDLING
For analytical questions involving multiple metrics:
- Identify all required metrics first and compute them in the same query.
- Use GROUP BY for the analytical dimension (e.g., product category, region, route).
- Use HAVING to filter low-sample results.
- Use ORDER BY to rank results.
- Use LIMIT 10 for ranking queries.

OPTIMIZATION RULES
- Avoid SELECT DISTINCT unless necessary.
- Avoid multiple scans of the same tables.
- Prefer aggregation queries over raw row outputs.
- Use LEFT JOIN when optional relationships exist.
- Ensure every non-aggregated column appears in GROUP BY.

OUTPUT FORMAT
Return ONLY the final SQL query.
Do not generate multiple queries.
Do not include explanations or comments.
"""


class SQLAgent:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or FALLBACK_MODELS[0]
        self.llm = _build_llm(self.model_name)
        self.schema: str = ""
        self.conversation_history: list = []  # For follow-up support
        self.current_session_id: Optional[str] = None

    def _invoke_llm(self, messages: list) -> str:
        """Invoke LLM with automatic fallback on a sequence of models."""
        last_error = None
        # Try models in sequence starting from the current one
        current_idx = 0
        if self.model_name in FALLBACK_MODELS:
            current_idx = FALLBACK_MODELS.index(self.model_name)
        
        for i in range(current_idx, len(FALLBACK_MODELS)):
            model = FALLBACK_MODELS[i]
            try:
                # Build temporary LLM if not using the primary one
                active_llm = self.llm if model == self.model_name else _build_llm(model)
                response = active_llm.invoke(messages)
                return response.content
            except Exception as e:
                last_error = e
                if "rate_limit_exceeded" in str(e).lower() and i < len(FALLBACK_MODELS) - 1:
                    logger.warning(f"Model {model} rate limited. Trying fallback: {FALLBACK_MODELS[i+1]}")
                    continue
                raise e
        
        if last_error:
            raise last_error
        return ""

    def refresh_schema(self):
        self.schema = get_semantic_schema_string()

    def _get_system_message(self) -> SystemMessage:
        schema = self.schema or get_semantic_schema_string()
        return SystemMessage(content=SYSTEM_PROMPT.format(schema=schema))

    def generate_sql(self, question: str) -> str:
        """Generate SQL from a natural language question."""
        schema = self.schema or get_semantic_schema_string()
        table_names = get_table_names()

        prompt = f"""User question: {question}

Database Schema:
{schema}

Generate the correct PostgreSQL SQL query. Only return the SQL query, no explanation."""

        messages = [
            self._get_system_message(),
        ] + self.conversation_history + [
            HumanMessage(content=prompt)
        ]

        content = self._invoke_llm(messages)
        sql = clean_sql(content)

        # Store in conversation history
        self.conversation_history.append(HumanMessage(content=f"Question: {question}"))
        self.conversation_history.append(AIMessage(content=sql))

        # Keep history bounded
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return sql

    async def load_history_from_db(self, session_id: str):
        """Load past questions and SQL answers from the database for context."""
        from database.db_connection import get_async_engine
        engine = get_async_engine()
        self.conversation_history = []
        self.current_session_id = session_id
        async with engine.connect() as conn:
            result = await conn.execute(text(
                "SELECT user_question, generated_sql FROM chats WHERE session_id = :sid ORDER BY created_at ASC LIMIT 10"
            ), {"sid": session_id})
            for row in result.fetchall():
                self.conversation_history.append(HumanMessage(content=f"Question: {row.user_question}"))
                self.conversation_history.append(AIMessage(content=row.generated_sql))

    async def correct_sql(self, original_sql: str, error: str, question: str) -> str:
        """Ask LLM to fix a failing SQL query."""
        schema = self.schema or get_semantic_schema_string()
        correction_prompt = f"""The SQL query you generated failed with an error.

Original question: {question}

Failed SQL:
{original_sql}

Error message:
{error}

Database Schema:
{schema}

Please fix the SQL query. Return only the corrected SQL, no explanation."""

        messages = [self._get_system_message(), HumanMessage(content=correction_prompt)]
        content = self._invoke_llm(messages)
        return clean_sql(content)

    async def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return a Pandas DataFrame with a strict hard limit asynchronously."""
        from database.db_connection import get_async_engine
        engine = get_async_engine()
        async with engine.connect() as conn:
            result = await conn.execute(text(sql))
            rows = result.fetchmany(1000)
            df = pd.DataFrame(rows, columns=list(result.keys()))
        return df

    async def run(self, question: str, session_id: str = None) -> Tuple[str, pd.DataFrame, str]:
        """
        Full pipeline: question -> SQL -> validate -> execute (with retry) -> DataFrame.
        Returns (final_sql, dataframe, error_or_empty_string).
        """
        # Load memory or handle session change
        if session_id:
            if session_id != self.current_session_id:
                # Session changed: clear and reload
                await self.load_history_from_db(session_id)
            elif not self.conversation_history:
                # Same session but history lost (rare): reload
                await self.load_history_from_db(session_id)

        schema = self.schema or get_semantic_schema_string()
        table_names = get_table_names()

        sql = self.generate_sql(question)
        last_error = ""

        for attempt in range(MAX_RETRY):
            # Validate
            valid, val_error = validate_sql(sql, table_names)
            if not valid:
                if attempt < MAX_RETRY - 1:
                    sql = await self.correct_sql(sql, val_error, question)
                    continue
                else:
                    return sql, pd.DataFrame(), f"Validation error: {val_error}"

            # Execute
            try:
                df = await self.execute_query(sql)
                return sql, df, ""
            except Exception as e:
                last_error = str(e)
                if attempt < MAX_RETRY - 1:
                    sql = await self.correct_sql(sql, last_error, question)
                else:
                    return sql, pd.DataFrame(), f"Execution error after {MAX_RETRY} attempts: {last_error}"

        return sql, pd.DataFrame(), last_error

    def clear_history(self):
        """Reset conversation history."""
        self.conversation_history = []


# Singleton instance
_agent_instance: Optional[SQLAgent] = None


def get_agent() -> SQLAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = SQLAgent()
    return _agent_instance

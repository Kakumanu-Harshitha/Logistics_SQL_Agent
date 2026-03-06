import pytest
from agents.sql_agent import get_agent
from utils.query_validator import validate_sql
import sqlglot

# Set of benchmark questions designed to test complex aggregation, limits, and joins
BENCHMARK_QUESTIONS = [
    "Which delivery routes have the highest delay rate?",
    "Which product categories generate the highest revenue?",
    "Which regions have the worst delivery performance?",
    "Show me the top 5 customers who order the most.",
    "Which cities have the most canceled deliveries?",
    "What is the average shipping distance per product department?",
    "Identify the drivers with the highest late delivery risk rates.",
    "What shipping mode is most popular for large orders?",
    "Compare the average order value between First Class and Standard Class shipping.",
    "Which warehouses handle the most orders with late delivery risk?"
]

@pytest.mark.asyncio
async def test_agent_generates_valid_sql_syntax():
    """Verify that the agent generates syntactically valid SQL according to sqlglot."""
    agent = get_agent()
    # We only test the first 3 to save time/tokens during basic CI
    for question in BENCHMARK_QUESTIONS[:3]:
        sql = agent.generate_sql(question)
        
        # Test 1: Should pass our internal validator
        is_valid, err = validate_sql(sql)
        assert is_valid is True, f"Failed internal validation: {err} | SQL: {sql}"
        
        # Test 2: Should parse cleanly via sqlglot
        try:
            parsed = sqlglot.parse_one(sql, dialect="postgres")
            assert parsed is not None
        except Exception as e:
            pytest.fail(f"sqlglot failed to parse generated SQL: {e} | SQL: {sql}")

@pytest.mark.asyncio
async def test_agent_enforces_aggregation_and_limits():
    """Verify that ranking questions include GROUP BY and LIMIT."""
    agent = get_agent()
    ranking_question = "Which product categories generate the highest revenue?"
    sql = agent.generate_sql(ranking_question)
    
    parsed = sqlglot.parse_one(sql, dialect="postgres")
    
    # Needs to be an aggregation query (should have GROUP BY)
    has_group_by = parsed.args.get("group") is not None
    assert has_group_by, f"Query for '{ranking_question}' missing GROUP BY clause. SQL: {sql}"
    
    # Needs to have a limit
    has_limit = parsed.args.get("limit") is not None
    assert has_limit, f"Query for '{ranking_question}' missing LIMIT clause. SQL: {sql}"

@pytest.mark.asyncio
async def test_agent_single_query_execution():
    """Ensure the agent outputs ONLY one query string."""
    agent = get_agent()
    sql = agent.generate_sql("Which delivery routes have the highest delay rate?")
    
    # sqlglot.parse returns a list of expressions. If there's more than 1, it hallucinated multiple queries.
    expressions = sqlglot.parse(sql, dialect="postgres")
    assert len(expressions) == 1, f"Agent generated multiple SQL statements instead of ONE. SQL: {sql}"

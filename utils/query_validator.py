"""
utils/query_validator.py
Validates LLM-generated SQL for safety and schema correctness.
"""
import re
from typing import Tuple
import sqlglot


BLOCKED_KEYWORDS = [
    r"\bDROP\b", r"\bDELETE\b", r"\bALTER\b", r"\bTRUNCATE\b",
    r"\bINSERT\b", r"\bUPDATE\b", r"\bCREATE\b", r"\bGRANT\b",
    r"\bREVOKE\b", r"\bEXECUTE\b", r"\bEXEC\b",
]

KNOWN_TABLES = [
    "customers", "orders", "deliveries", "drivers",
    "routes", "warehouses", "products",
]


def validate_sql(sql: str, schema_tables: list = None) -> Tuple[bool, str]:
    """
    Validate an SQL query using sqlglot AST parsing for strict read-only safety.
    Returns (is_valid, error_message).
    """
    if not sql or not sql.strip():
        return False, "Empty SQL query."

    try:
        # Parse the SQL using sqlglot for PostgreSQL dialect
        parsed = sqlglot.parse(sql, dialect="postgres")
    except sqlglot.errors.ParseError as e:
        return False, f"SQL Syntax Error: {str(e)}"

    if not parsed:
        return False, "Could not parse SQL."

    if len(parsed) > 1:
        return False, "Multiple SQL statements detected. Only one single statement is allowed."

    ast = parsed[0]

    if not ast:
         return False, "Empty parsed statement."

    # Rule 1: Must be a SELECT (or WITH CTE containing a SELECT)
    if not isinstance(ast, sqlglot.exp.Select):
        return False, f"Only SELECT queries are permitted. Found: {ast.key.upper() if hasattr(ast, 'key') else type(ast).__name__}"

    # Rule 2: Walk the AST and actively reject any blocked statements (just in case they are nested)
    blocked_expressions = (
        sqlglot.exp.Drop, sqlglot.exp.Delete, sqlglot.exp.Update, 
        sqlglot.exp.Insert, sqlglot.exp.Alter, sqlglot.exp.Command,
        sqlglot.exp.Commit
    )

    for node in ast.walk():
        # Check against blocked generic definitions
        if isinstance(node, blocked_expressions):
            return False, f"Blocked operation detected: {node.key.upper() if hasattr(node, 'key') else type(node).__name__}"

    # Check for recognized tables
    tables = schema_tables or KNOWN_TABLES
    found_table = False
    
    # Extract all tables from the AST
    for table_node in ast.find_all(sqlglot.exp.Table):
        table_name = table_node.name.lower()
        if table_name in tables:
            found_table = True
            break
            
    if not found_table:
        return False, f"No recognized table found in query. Known tables: {', '.join(KNOWN_TABLES)}"

    return True, ""


def clean_sql(sql: str) -> str:
    """
    Clean LLM-generated SQL: strip markdown code fences,
    leading/trailing whitespace, and trailing semicolons.
    """
    sql = sql.strip()
    # Remove markdown fences
    if sql.startswith("```"):
        lines = sql.split("\n")
        # Remove first line (```sql) and last line (```)
        lines = lines[1:] if lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
        sql = "\n".join(lines).strip()
    # Remove trailing semicolons for safety
    sql = sql.rstrip(";").strip()
    return sql

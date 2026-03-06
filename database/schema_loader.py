"""
database/schema_loader.py
Loads database schema dynamically for LLM context.
"""
from inspect import signature
from sqlalchemy import inspect, text
from database.db_connection import get_engine


def get_schema_string() -> str:
    """
    Introspect PostgreSQL and return a formatted schema string
    that is passed to the LLM before SQL generation.
    """
    engine = get_engine()
    inspector = inspect(engine)
    tables = inspector.get_table_names(schema="public")

    schema_parts = []
    for table_name in sorted(tables):
        columns = inspector.get_columns(table_name, schema="public")
        fks = inspector.get_foreign_keys(table_name, schema="public")
        col_defs = []
        for col in columns:
            nullable = "" if col.get("nullable", True) else " NOT NULL"
            col_defs.append(f"  {col['name']} {col['type']}{nullable}")
        schema_parts.append(f"Table: {table_name}\nColumns:\n" + "\n".join(col_defs))
        if fks:
            for fk in fks:
                ref = f"  FK: {fk['constrained_columns']} -> {fk['referred_table']}({fk['referred_columns']})"
                schema_parts.append(ref)
        schema_parts.append("")

    return "\n".join(schema_parts)


def get_schema_dict() -> dict:
    """Return schema as a dictionary {table: [columns]}."""
    engine = get_engine()
    inspector = inspect(engine)
    tables = inspector.get_table_names(schema="public")
    schema = {}
    for table in sorted(tables):
        columns = inspector.get_columns(table, schema="public")
        schema[table] = [
            {"name": col["name"], "type": str(col["type"]), "nullable": col.get("nullable", True)}
            for col in columns
        ]
    return schema


def get_table_names() -> list:
    """Return list of table names."""
    engine = get_engine()
    inspector = inspect(engine)
    return inspector.get_table_names(schema="public")


def get_column_names(table: str) -> list:
    """Return column names for a specific table."""
    engine = get_engine()
    inspector = inspect(engine)
    columns = inspector.get_columns(table, schema="public")
    return [col["name"] for col in columns]

SEMANTIC_CONTEXT = {
    "deliveries": {
        "delivery_id": "Primary key for a delivery.",
        "order_id": "Joins with orders.order_id.",
        "driver_id": "Joins with drivers.driver_id.",
        "route_id": "Joins with routes.route_id.",
        "warehouse_id": "Joins with warehouses.warehouse_id.",
        "delivery_status": "Current status of delivery (e.g., Shipping on time, Late delivery).",
        "late_delivery_risk": "1 if delivery is high risk/late, 0 otherwise."
    },
    "orders": {
        "order_id": "Primary key for an order.",
        "customer_id": "Joins with customers.customer_id.",
        "product_id": "Joins with products.product_id.",
        "order_value": "Total revenue/sales value of the order.",
        "order_status": "Status of the order (e.g., COMPLETE, PENDING)."
    },
    "routes": {
        "route_id": "Unique route identifier. Joins with deliveries.route_id.",
        "origin_city": "City where shipment begins.",
        "destination_city": "City where shipment ends.",
        "shipping_mode": "Method of shipping (e.g., Standard Class, First Class).",
        "distance_km": "Total distance of the route in kilometers."
    },
    "products": {
        "product_id": "Primary key for a product. Joins with orders.product_id.",
        "product_name": "Name of the product.",
        "category_name": "Category the product belongs to (e.g., Cleats, Women's Apparel).",
        "product_price": "Price per unit."
    },
    "customers": {
        "customer_id": "Primary key for a customer. Joins with orders.customer_id.",
        "customer_segment": "Segment of the customer (e.g., Consumer, Corporate)."
    },
    "drivers": {
        "driver_id": "Primary key for a driver. Joins with deliveries.driver_id.",
        "rating": "Performance rating of the driver (1.0 to 5.0)."
    },
    "warehouses": {
        "warehouse_id": "Primary key for a warehouse. Joins with deliveries.warehouse_id.",
        "warehouse_city": "City where the warehouse is located."
    }
}

def get_semantic_schema_string() -> str:
    """
    Introspect PostgreSQL and return a formatted schema string
    combined with semantic business descriptions for the LLM.
    """
    engine = get_engine()
    inspector = inspect(engine)
    tables = inspector.get_table_names(schema="public")

    schema_parts = []
    for table_name in sorted(tables):
        columns = inspector.get_columns(table_name, schema="public")
        fks = inspector.get_foreign_keys(table_name, schema="public")
        
        col_defs = []
        for col in columns:
            nullable = "" if col.get("nullable", True) else " NOT NULL"
            col_name = col['name']
            col_type = col['type']
            
            # Inject semantics if available
            semantic_desc = ""
            if table_name in SEMANTIC_CONTEXT and col_name in SEMANTIC_CONTEXT[table_name]:
                semantic_desc = f" -- {SEMANTIC_CONTEXT[table_name][col_name]}"
                
            col_defs.append(f"  {col_name} {col_type}{nullable}{semantic_desc}")
            
        schema_parts.append(f"Table: {table_name}\nColumns:\n" + "\n".join(col_defs))
        
        if fks:
            for fk in fks:
                ref = f"  FK: {fk['constrained_columns']} -> {fk['referred_table']}({fk['referred_columns']})"
                schema_parts.append(ref)
        schema_parts.append("")

    return "\n".join(schema_parts)

"""
database/create_tables.py
Creates all tables in PostgreSQL for the DataCo Supply Chain dataset.
Run this ONCE before ingesting data.
"""
import os
from sqlalchemy import text
from backend.database.db_connection import get_engine
import sys

# DDL same as above
DDL = """
-- Drop existing tables in reverse dependency order
DROP TABLE IF EXISTS deliveries CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
DROP TABLE IF EXISTS drivers CASCADE;
DROP TABLE IF EXISTS routes CASCADE;
DROP TABLE IF EXISTS warehouses CASCADE;
DROP TABLE IF EXISTS products CASCADE;

-- Products table
CREATE TABLE products (
    product_id        INTEGER PRIMARY KEY,
    product_name      VARCHAR(255),
    category_name     VARCHAR(100),
    department_name   VARCHAR(100),
    product_price     NUMERIC(12,2),
    product_status    INTEGER DEFAULT 0
);

-- Warehouses / department stores
CREATE TABLE warehouses (
    warehouse_id      SERIAL PRIMARY KEY,
    warehouse_city    VARCHAR(100),
    department_name   VARCHAR(100),
    market            VARCHAR(50),
    capacity          INTEGER DEFAULT 10000,
    manager           VARCHAR(100)
);

-- Routes
CREATE TABLE routes (
    route_id              SERIAL PRIMARY KEY,
    origin_city           VARCHAR(100),
    destination_city      VARCHAR(100),
    shipping_mode         VARCHAR(50),
    average_traffic_level VARCHAR(20),
    distance_km           NUMERIC(10,2)
);

-- Drivers (delivery agents)
CREATE TABLE drivers (
    driver_id         VARCHAR(50) PRIMARY KEY,
    driver_name       VARCHAR(100),
    experience_years  INTEGER,
    vehicle_type      VARCHAR(50),
    rating            NUMERIC(3,2)
);

-- Customers
CREATE TABLE customers (
    customer_id       INTEGER PRIMARY KEY,
    customer_name     VARCHAR(100),
    city              VARCHAR(100),
    region            VARCHAR(100),
    country           VARCHAR(100),
    customer_segment  VARCHAR(50),
    signup_date       DATE
);

-- Orders
CREATE TABLE orders (
    order_id          INTEGER PRIMARY KEY,
    customer_id       INTEGER REFERENCES customers(customer_id),
    order_date        TIMESTAMP,
    product_id        INTEGER REFERENCES products(product_id),
    order_value       NUMERIC(12,2),
    order_status      VARCHAR(30),
    shipping_mode     VARCHAR(50),
    order_region      VARCHAR(100),
    order_city        VARCHAR(100),
    order_country     VARCHAR(100),
    benefit_per_order NUMERIC(10,2),
    order_item_discount NUMERIC(10,2),
    order_item_quantity INTEGER
);

-- Deliveries
CREATE TABLE deliveries (
    delivery_id       SERIAL PRIMARY KEY,
    order_id          INTEGER REFERENCES orders(order_id),
    driver_id         VARCHAR(50) REFERENCES drivers(driver_id),
    route_id          INTEGER REFERENCES routes(route_id),
    warehouse_id      INTEGER REFERENCES warehouses(warehouse_id),
    pickup_time       TIMESTAMP,
    delivery_time     TIMESTAMP,
    delivery_status   VARCHAR(30),
    distance_km       NUMERIC(10,2),
    days_for_shipping_real      INTEGER,
    days_for_shipment_scheduled INTEGER,
    late_delivery_risk          INTEGER
);

-- Chats (Persistent Memory)
CREATE TABLE chats (
    id SERIAL PRIMARY KEY,
    session_id TEXT,
    user_question TEXT,
    generated_sql TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def create_tables():
    engine = get_engine()
    
    with engine.begin() as conn:
        statements = [s.strip() for s in DDL.split(';') if s.strip()]
        for statement in statements:
            conn.execute(text(statement))
            
    print("✅ All tables created successfully.")

if __name__ == "__main__":
    create_tables()

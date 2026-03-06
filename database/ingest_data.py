"""
database/ingest_data.py
Reads DataCoSupplyChainDataset.csv and maps it into the normalized tables.
Run AFTER create_tables.py.
"""
import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import text
from database.db_connection import get_engine
from datetime import datetime, timedelta
import random

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "DataCoSupplyChainDataset.csv")


def safe_int(val, default=0):
    try:
        v = int(float(val))
        return v if not np.isnan(float(val)) else default
    except:
        return default


def safe_float(val, default=0.0):
    try:
        v = float(val)
        return v if not np.isnan(v) else default
    except:
        return default


def ingest():
    print("📂 Reading CSV...")
    df = pd.read_csv(CSV_PATH, encoding="latin-1", low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"   Rows: {len(df)}, Cols: {len(df.columns)}")

    engine = get_engine()

    # ------------------------------------------------------------
    # 1. PRODUCTS
    # ------------------------------------------------------------
    print("📦 Ingesting products...")
    prod_cols = {
        "Product Card Id": "product_id",
        "Product Name": "product_name",
        "Category Name": "category_name",
        "Department Name": "department_name",
        "Product Price": "product_price",
        "Product Status": "product_status",
    }
    prod_df = df[list(prod_cols.keys())].rename(columns=prod_cols).drop_duplicates(subset=["product_id"])
    prod_df["product_id"] = prod_df["product_id"].apply(safe_int)
    prod_df["product_price"] = prod_df["product_price"].apply(safe_float)
    prod_df["product_status"] = prod_df["product_status"].apply(lambda x: safe_int(x, 0))
    prod_df = prod_df[prod_df["product_id"] > 0]
    prod_df.to_sql("products", engine, if_exists="append", index=False, method="multi", chunksize=500)
    print(f"   ✅ {len(prod_df)} products")

    # ------------------------------------------------------------
    # 2. WAREHOUSES  (from Department / Market)
    # ------------------------------------------------------------
    print("🏭 Ingesting warehouses...")
    wh_df = df[["Department Name", "Market", "Order City"]].drop_duplicates().reset_index(drop=True)
    wh_df.columns = ["department_name", "market", "warehouse_city"]
    wh_df["warehouse_id"] = range(1, len(wh_df) + 1)
    wh_df["capacity"] = [random.randint(5000, 20000) for _ in range(len(wh_df))]
    wh_df["manager"] = [f"Manager_{i}" for i in range(1, len(wh_df) + 1)]
    wh_df = wh_df[["warehouse_id", "warehouse_city", "department_name", "market", "capacity", "manager"]]
    wh_df.to_sql("warehouses", engine, if_exists="append", index=False, method="multi", chunksize=500)
    print(f"   ✅ {len(wh_df)} warehouses")
    wh_lookup = wh_df.set_index(["department_name", "market", "warehouse_city"])["warehouse_id"].to_dict()

    # ------------------------------------------------------------
    # 3. ROUTES  (from Shipping Mode + Region)
    # ------------------------------------------------------------
    print("🗺️  Ingesting routes...")
    route_df = df[["Shipping Mode", "Order Region", "Order City"]].drop_duplicates().reset_index(drop=True)
    route_df.columns = ["shipping_mode", "destination_city", "origin_city"]
    route_df["route_id"] = range(1, len(route_df) + 1)
    traffic_map = {"Standard Class": "Medium", "First Class": "Low", "Second Class": "High", "Same Day": "Very High"}
    route_df["average_traffic_level"] = route_df["shipping_mode"].map(traffic_map).fillna("Medium")
    route_df["distance_km"] = [round(random.uniform(50, 1500), 2) for _ in range(len(route_df))]
    route_df = route_df[["route_id", "origin_city", "destination_city", "shipping_mode", "average_traffic_level", "distance_km"]]
    route_df.to_sql("routes", engine, if_exists="append", index=False, method="multi", chunksize=500)
    print(f"   ✅ {len(route_df)} routes")
    route_lookup = route_df.set_index(["shipping_mode", "destination_city", "origin_city"])["route_id"].to_dict()

    # ------------------------------------------------------------
    # 4. DRIVERS (synthetic from order count patterns)
    # ------------------------------------------------------------
    print("🚗 Ingesting drivers...")
    num_drivers = 500
    vehicle_types = ["Motorcycle", "Van", "Truck", "Bicycle", "Car"]
    driver_records = []
    for i in range(1, num_drivers + 1):
        driver_records.append({
            "driver_id": f"DRV{i:04d}",
            "driver_name": f"Driver_{i}",
            "experience_years": random.randint(1, 15),
            "vehicle_type": random.choice(vehicle_types),
            "rating": round(random.uniform(2.5, 5.0), 2),
        })
    driver_df = pd.DataFrame(driver_records)
    driver_df.to_sql("drivers", engine, if_exists="append", index=False, method="multi", chunksize=500)
    driver_ids = driver_df["driver_id"].tolist()
    print(f"   ✅ {len(driver_df)} drivers")

    # ------------------------------------------------------------
    # 5. CUSTOMERS
    # ------------------------------------------------------------
    print("👤 Ingesting customers...")
    cust_cols = {
        "Customer Id": "customer_id",
        "Customer Fname": "customer_name",
        "Customer City": "city",
        "Order Region": "region",
        "Customer Country": "country",
        "Customer Segment": "customer_segment",
    }
    cust_df = df[list(cust_cols.keys())].rename(columns=cust_cols).copy()
    cust_df["customer_name"] = cust_df["customer_name"].fillna("Unknown")
    cust_df["customer_id"] = cust_df["customer_id"].apply(safe_int)
    cust_df = cust_df.drop_duplicates(subset=["customer_id"])
    cust_df = cust_df[cust_df["customer_id"] > 0]
    # Synthesize signup dates
    base_date = datetime(2015, 1, 1)
    cust_df["signup_date"] = [
        (base_date + timedelta(days=random.randint(0, 3 * 365))).date()
        for _ in range(len(cust_df))
    ]
    cust_df.to_sql("customers", engine, if_exists="append", index=False, method="multi", chunksize=500)
    print(f"   ✅ {len(cust_df)} customers")
    valid_customer_ids = set(cust_df["customer_id"].tolist())

    # ------------------------------------------------------------
    # 6. ORDERS
    # ------------------------------------------------------------
    print("📋 Ingesting orders...")
    order_df = df[[
        "Order Id", "Order Customer Id", "order date (DateOrders)",
        "Product Card Id", "Sales", "Order Status", "Shipping Mode",
        "Order Region", "Order City", "Order Country",
        "Benefit per order", "Order Item Discount", "Order Item Quantity"
    ]].copy()
    order_df.columns = [
        "order_id", "customer_id", "order_date", "product_id",
        "order_value", "order_status", "shipping_mode", "order_region",
        "order_city", "order_country", "benefit_per_order",
        "order_item_discount", "order_item_quantity"
    ]
    order_df["order_id"] = order_df["order_id"].apply(safe_int)
    order_df["customer_id"] = order_df["customer_id"].apply(safe_int)
    order_df["product_id"] = order_df["product_id"].apply(safe_int)
    order_df["order_value"] = order_df["order_value"].apply(safe_float)
    order_df["benefit_per_order"] = order_df["benefit_per_order"].apply(safe_float)
    order_df["order_item_discount"] = order_df["order_item_discount"].apply(safe_float)
    order_df["order_item_quantity"] = order_df["order_item_quantity"].apply(safe_int)
    order_df["order_date"] = pd.to_datetime(order_df["order_date"], errors="coerce")

    # Filter to valid references
    valid_product_ids = set(prod_df["product_id"].tolist())
    order_df = order_df[
        (order_df["order_id"] > 0) &
        (order_df["customer_id"].isin(valid_customer_ids)) &
        (order_df["product_id"].isin(valid_product_ids))
    ]
    order_df = order_df.drop_duplicates(subset=["order_id"])
    order_df.to_sql("orders", engine, if_exists="append", index=False, method="multi", chunksize=500)
    print(f"   ✅ {len(order_df)} orders")
    valid_order_ids = set(order_df["order_id"].tolist())

    # ------------------------------------------------------------
    # 7. DELIVERIES
    # ------------------------------------------------------------
    print("🚚 Ingesting deliveries...")
    del_df = df[[
        "Order Id", "Shipping Mode", "Order Region", "Order City",
        "Department Name", "Market",
        "Delivery Status", "Days for shipping (real)",
        "Days for shipment (scheduled)", "Late_delivery_risk",
        "shipping date (DateOrders)", "order date (DateOrders)"
    ]].copy()
    del_df.columns = [
        "order_id", "shipping_mode", "destination_city", "origin_city",
        "department_name", "market",
        "delivery_status", "days_for_shipping_real",
        "days_for_shipment_scheduled", "late_delivery_risk",
        "shipping_date", "order_date"
    ]
    del_df["order_id"] = del_df["order_id"].apply(safe_int)
    del_df = del_df[del_df["order_id"].isin(valid_order_ids)]

    # Map route_id and warehouse_id
    del_df["route_id"] = del_df.apply(
        lambda r: route_lookup.get((r["shipping_mode"], r["destination_city"], r["origin_city"]), 1), axis=1
    )
    del_df["warehouse_id"] = del_df.apply(
        lambda r: wh_lookup.get((r["department_name"], r["market"], r["origin_city"]), 1), axis=1
    )
    del_df["driver_id"] = [driver_ids[i % num_drivers] for i in range(len(del_df))]
    del_df["order_date"] = pd.to_datetime(del_df["order_date"], errors="coerce")
    del_df["shipping_date"] = pd.to_datetime(del_df["shipping_date"], errors="coerce")
    del_df["pickup_time"] = del_df["order_date"] + pd.to_timedelta(
        del_df["days_for_shipping_real"].apply(lambda x: safe_int(x, 1)) - 1, unit="D"
    )
    del_df["delivery_time"] = del_df["shipping_date"]
    del_df["days_for_shipping_real"] = del_df["days_for_shipping_real"].apply(safe_int)
    del_df["days_for_shipment_scheduled"] = del_df["days_for_shipment_scheduled"].apply(safe_int)
    del_df["late_delivery_risk"] = del_df["late_delivery_risk"].apply(lambda x: safe_int(x, 0))
    del_df["distance_km"] = del_df["route_id"].map(route_df.set_index("route_id")["distance_km"])

    final_del = del_df[[
        "order_id", "driver_id", "route_id", "warehouse_id",
        "pickup_time", "delivery_time", "delivery_status", "distance_km",
        "days_for_shipping_real", "days_for_shipment_scheduled", "late_delivery_risk"
    ]]
    final_del.to_sql("deliveries", engine, if_exists="append", index=False, method="multi", chunksize=500)
    print(f"   ✅ {len(final_del)} deliveries")

    print("\n🎉 Data ingestion complete!")


if __name__ == "__main__":
    ingest()

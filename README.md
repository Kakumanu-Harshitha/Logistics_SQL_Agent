# 🚚 Logistics AI Analytics SQL Agent

An AI-powered logistics analytics platform using **LangChain + Groq (Llama 3.3 70B) + PostgreSQL + Streamlit**. The system converts natural language questions into optimized SQL queries, executes them against a PostgreSQL database containing supply chain data (orders, deliveries, routes, customers, etc.), and generates insights and visualizations.

## 📊 Dataset
**DataCo Smart Supply Chain Dataset** — 180,000+ supply chain records covering orders, deliveries, customers, products, shipping modes, and regions.

## 🏗️ Architecture

```mermaid
flowchart TD
    User([User]) -->|Natural Language Question| UI[Streamlit Frontend]
    UI -->|POST /query| Backend[FastAPI Backend]
    
    subgraph Backend Services
        Backend --> SQL_Agent[LangChain SQL Agent\nGroq Llama 3.3 70B]
        SQL_Agent -->|Generates ONE Optimized Query| Validator[SQL Validator]
        Validator --> DB_Execution[PostgreSQL Database]
        DB_Execution -->|Results DataFrame| Data_Processing[Pandas Data Processing]
        
        Data_Processing --> Visualizer[Plotly Visualizer]
        Data_Processing --> Insight_Agent[LangChain Insight Generator]
    end
    
    Visualizer -->|JSON Chart| Backend
    Insight_Agent -->|Business Insight| Backend
    Backend -->|Response Object| UI
    UI -->|Displays Data, Chart, Insight| User
```

### Core Components
1. **Frontend (Streamlit)**: Interactive UI for answering natural language questions, displaying query history, interactive charts, and business insights. Also features an XGBoost-powered predictive model for delivery delays.
2. **Backend API (FastAPI)**: REST endpoints for querying, model training, and predictions.
3. **Core SQL Agent (LangChain + Groq)**: Context-aware agent running on Llama-3.3-70b. It generates highly optimized, single-shot PostgreSQL analytics queries (using strict aggregations, proper JOINs, limits, window functions, and early filtering) without any intermediate exploratory queries.
4. **Data Layer (PostgreSQL)**: Fully normalized relational database with tables for products, orders, deliveries, customers, routes, drivers, and warehouses.

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=postgresql://postgres:password@localhost:5432/logistics_db
```

### 3. Setup Database
Ensure PostgreSQL is running locally, then create tables and ingest data:
```powershell
# Create tables
python -m database.create_tables

# Ingest DataCo CSV data
python -m database.ingest_data
```

### 4. Start Services
Start the **FastAPI Backend**:
```powershell
# Opens on port 8000
uvicorn main:app --reload --port 8000
```

Start the **Streamlit Frontend** (in a new terminal):
```powershell
# Opens on port 8501
streamlit run app.py
```

Navigate to **http://localhost:8501** in your browser.

## 💡 Example Questions

- _"Which delivery routes have the highest delay rate?"_
- _"Which product categories generate the most revenue but also experience the highest late delivery risk?"_
- _"What is the average delivery time per city?"_
- _"Which shipping mode has the worst late delivery rate?"_
- _"Which regions have the most canceled orders?"_

## 📂 Project Structure

```text
logistics_sql_agent/
├── app.py                   # Streamlit frontend
├── main.py                  # FastAPI entry point
├── requirements.txt         # Python dependencies
├── .env                     # Credentials (git-ignored)
├── data/
│   └── DataCoSupplyChainDataset.csv
├── api/
│   └── routes.py            # FastAPI endpoints
├── agents/
│   ├── sql_agent.py         # Prompt engineering & execution for SQL Agent
│   └── query_planner.py     # Request parsing and planning
├── database/
│   ├── db_connection.py     # SQLAlchemy engine setup
│   ├── create_tables.py     # DDL script for DB normalization
│   └── ingest_data.py       # Data mapping & CSV ingestion script
├── analytics/
│   ├── insights.py          # LangChain-powered business insights
│   └── visualizations.py    # Auto-generates Plotly charts
├── ml/
│   └── delay_prediction.py  # XGBoost delivery delay predictor
└── utils/
    └── query_validator.py   # SQL syntax and table validation
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | API and Database Health check |
| `GET` | `/schema` | Introspect database schema |
| `POST` | `/query` | Processes Natural Language → SQL → Results |
| `GET` | `/sample-questions` | Retrieves example questions |
| `POST` | `/predict-delay` | ML Delay prediction inference |
| `POST` | `/train-model` | Trains the XGBoost delay model |
| `POST` | `/clear-history` | Resets the conversation memory |

# 🚚 Logistics AI Analytics SQL Agent

An AI-powered logistics analytics platform using **LangChain + Groq (Llama 3.3 70B) + PostgreSQL + Streamlit**. The system converts natural language questions into optimized SQL queries, executes them against a PostgreSQL database containing supply chain data (orders, deliveries, customers, products, shipping modes, and regions).

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
1. **Frontend (Streamlit)**: Modern AI dashboard UI. Features live metric cards, dynamic query input, vertical results flow (SQL -> Table -> Chart -> Insight), and an XGBoost-powered predictive model for delivery delays.
2. **Backend API (FastAPI)**: REST endpoints for querying, health checks, global dashboard metrics, and ML predictions.
3. **Core SQL Agent (LangChain + Groq)**: Context-aware agent running on Llama-3.3-70b. It generates highly optimized, single-shot PostgreSQL analytics queries.
4. **Data Layer (PostgreSQL)**: Normalized database with tables for products, orders, deliveries, customers, routes, and drivers.

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
python -m backend.database.create_tables

# Ingest DataCo CSV data
python -m backend.database.ingest_data
```

### 4. Start Services
Start the **FastAPI Backend**:
```powershell
# Opens on port 8000
uvicorn backend.main:app --reload --port 8000
```

Start the **Streamlit Frontend** (in a new terminal):
```powershell
# Opens on default port 8501 or next available
streamlit run frontend/app.py
```

Navigate to **http://localhost:8501** in your browser.

## 💡 Example Questions

- _"Which delivery routes have the highest delay rate?"_
- _"Which product categories generate the most revenue?"_
- _"What is the average delivery time per city?"_
- _"Which shipping mode has the worst late delivery rate?"_
- _"Show me the top 10 customers by order value"_

## 🤖 ML Delay Prediction
The platform includes an **XGBoost-powered predictive engine** that estimates the probability of delivery delays based on real-time factors.

### Features:
- **Real-time Inference**: Predict delays using factors like distance, shipping mode, traffic level, and driver experience.
- **One-Click Training**: Re-train the model directly from the UI or via the `/train-model` API endpoint using current database records.
- **Accuracy Metrics**: View model performance (RMSE, MAE) after training.

### How to use:
1. Navigate to the **"Delivery Delay Prediction"** tab in the Sidebar.
2. Adjust the sliders and inputs for shipment parameters.
3. Click **"Predict Delay Probability"** to see the AI's estimation and confidence level.

## 📂 Project Structure

```text
logistics_sql_agent/
├── backend/                 # All backend logic
│   ├── main.py              # FastAPI entry point
│   ├── api/
│   │   └── routes.py        # FastAPI endpoints
│   ├── agents/
│   │   ├── sql_agent.py     # SQL generation logic
│   │   └── query_planner.py # Query planning logic
│   ├── database/
│   │   ├── db_connection.py # DB engine
│   │   ├── schema_loader.py # Schema introspection
│   │   ├── create_tables.py # DDL setup
│   │   └── ingest_data.py   # Data ingestion
│   ├── analytics/
│   │   ├── insights.py      # LLM insights
│   │   ├── visualizations.py# Plotly charts
│   │   └── dashboard.py     # Global metrics Calculation
│   ├── ml/
│   │   └── delay_prediction.py # ML model
│   ├── utils/
│   │   └── query_validator.py # SQL validation
│   └── __init__.py
├── frontend/
│   └── app.py               # Streamlit UI
├── data/                    # Datasets
├── tests/                   # Unit tests
├── requirements.txt         # Project dependencies
├── .env                     # Credentials (git-ignored)
└── README.md
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | API, Database, and Model status |
| `GET` | `/dashboard-metrics` | Dynamic global logistics metrics |
| `GET` | `/schema` | Introspect database schema |
| `POST` | `/query` | Natural Language → SQL → Results |
| `GET` | `/sample-questions` | Retrieves example questions |
| `POST` | `/predict-delay` | ML Delay prediction inference |
| `POST` | `/train-model` | Trains the XGBoost delay model |
| `POST` | `/clear-history` | Resets the conversation memory |
## Model Performance

The delivery delay prediction model was evaluated using standard classification metrics.

| Metric | Value | Description |
|------|------|-------------|
| Accuracy | **74.6%** | Overall correctness of predictions |
| Recall | **89.9%** | Ability to correctly identify late deliveries |
| F1-Score | **80.3%** | Balance between precision and recall |
| Optimal Threshold | **0.4939** | Best probability threshold for classification |

**Key Insight:**  
The model achieves a high **recall of 89.9%**, making it highly effective at detecting potential late deliveries and enabling proactive logistics planning.
# Owner
[Kakumanu Harshitha](https://github.com/Kakumanu-Harshitha)
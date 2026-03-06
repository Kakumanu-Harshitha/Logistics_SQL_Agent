"""
app.py
Streamlit frontend for AI-powered Logistics SQL Agent.
"""
import streamlit as st
import requests
import pandas as pd
import plotly.io as pio
import json
import time
import uuid

# ─── Configuration ──────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="🚚 Logistics AI Agent",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #111827 100%);
    border-right: 1px solid #1e3a5f;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 100%);
    border: 1px solid #2563eb44;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 8px 0;
    box-shadow: 0 4px 20px rgba(37,99,235,0.15);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #60a5fa;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

/* SQL code block */
.sql-block {
    background: #0d1b2a;
    border: 1px solid #2563eb55;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 16px;
    font-family: 'Courier New', monospace;
    font-size: 0.85rem;
    color: #93c5fd;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 250px;
    overflow-y: auto;
}

/* Insight block */
.insight-block {
    background: linear-gradient(135deg, #0c2a1f 0%, #0a2218 100%);
    border: 1px solid #059669;
    border-left: 4px solid #10b981;
    border-radius: 8px;
    padding: 16px;
    color: #a7f3d0;
    font-size: 0.95rem;
    line-height: 1.7;
}

/* Error block */
.error-block {
    background: #2d0a0a;
    border: 1px solid #ef4444;
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 16px;
    color: #fca5a5;
}

/* Steps */
.step-tag {
    display: inline-block;
    background: #1e3a5f;
    color: #93c5fd;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    margin: 2px;
    border: 1px solid #2563eb55;
}

/* Header */
.hero-header {
    text-align: center;
    padding: 20px 0 10px 0;
}
.hero-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #34d399, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-size: 200% auto;
    animation: gradient 3s linear infinite;
}
@keyframes gradient {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.hero-sub {
    color: #64748b;
    font-size: 1rem;
    margin-top: 4px;
}

/* Section headers */
.section-header {
    color: #60a5fa;
    font-size: 1rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px;
    margin: 16px 0 12px 0;
}

/* Badge */
.badge-green { color: #34d399; font-weight: 600; }
.badge-red   { color: #f87171; font-weight: 600; }
.badge-blue  { color: #60a5fa; font-weight: 600; }

/* Input box */
.stTextInput > div > div > input,
.stTextArea textarea {
    background: #0f1f35 !important;
    color: #e2e8f0 !important;
    border: 1px solid #2563eb77 !important;
    border-radius: 8px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px #3b82f633 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    box-shadow: 0 4px 15px rgba(37,99,235,0.4) !important;
    transform: translateY(-1px) !important;
}

/* History item */
.history-item {
    background: #0f1f35;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    cursor: pointer;
    font-size: 0.88rem;
    color: #94a3b8;
    transition: all 0.2s ease;
}
.history-item:hover {
    border-color: #3b82f6;
    color: #e2e8f0;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* Risk badge */
.risk-high {
    background: #3b0a0a; color: #f87171;
    border: 1px solid #ef4444;
    border-radius: 20px; padding: 6px 16px;
    font-weight: 700; font-size: 1.1rem;
    display: inline-block;
}
.risk-low {
    background: #052e16; color: #34d399;
    border: 1px solid #10b981;
    border-radius: 20px; padding: 6px 16px;
    font-weight: 700; font-size: 1.1rem;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────────────────────────
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "trigger_query" not in st.session_state:
    st.session_state.trigger_query = False


# ─── Helper Functions ────────────────────────────────────────────────────────────
def check_api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def get_sample_questions():
    try:
        r = requests.get(f"{API_BASE}/sample-questions", timeout=3)
        return r.json().get("questions", []) if r.status_code == 200 else []
    except Exception:
        return []


def get_schema():
    try:
        r = requests.get(f"{API_BASE}/schema", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def run_query(question: str, clear_history: bool = False):
    payload = {
        "question": question, 
        "clear_history": clear_history,
        "session_id": st.session_state.session_id
    }
    r = requests.post(f"{API_BASE}/query", json=payload, timeout=60)
    if r.status_code == 200:
        return r.json()
    else:
        return {"error": r.text, "sql": "", "insight": ""}


def predict_delay(features: dict):
    try:
        r = requests.post(f"{API_BASE}/predict-delay", json=features, timeout=15)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def train_model():
    try:
        r = requests.post(f"{API_BASE}/train-model", timeout=300)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-header">⚡ System Status</p>', unsafe_allow_html=True)

    health = check_api_health()
    if health:
        db_status = health.get("database", "unknown")
        model_trained = health.get("model_trained", False)
        st.markdown(
            f'<span class="badge-green">● API Online</span>',
            unsafe_allow_html=True
        )
        db_color = "badge-green" if db_status == "connected" else "badge-red"
        db_icon = "●" if db_status == "connected" else "○"
        st.markdown(
            f'<span class="{db_color}">{db_icon} Database {db_status.title()}</span>',
            unsafe_allow_html=True
        )
        ml_color = "badge-green" if model_trained else "badge-blue"
        ml_text = "ML Model Ready" if model_trained else "ML Model Not Trained"
        st.markdown(f'<span class="{ml_color}">● {ml_text}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-red">○ API Offline — Start backend first</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Sample Questions ────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">💡 Sample Questions</p>', unsafe_allow_html=True)
    sample_qs = get_sample_questions()
    for q in sample_qs:
        if st.button(q, key=f"sq_{hash(q)}", use_container_width=True):
            st.session_state["selected_question"] = q
            st.session_state["trigger_query"] = True
            st.rerun()

    st.markdown("---")

    # ── Schema Browser ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">🗄️ Database Schema</p>', unsafe_allow_html=True)
    schema_data = get_schema()
    if schema_data.get("schema_dict"):
        for table, cols in schema_data["schema_dict"].items():
            with st.expander(f"📋 {table} ({len(cols)} cols)"):
                for col in cols:
                    nullable = "" if col["nullable"] else " ★"
                    st.markdown(
                        f'<span style="color:#60a5fa">{col["name"]}</span> '
                        f'<span style="color:#475569;font-size:0.8rem">{col["type"]}{nullable}</span>',
                        unsafe_allow_html=True
                    )

    st.markdown("---")

    # ── Conversation Controls ───────────────────────────────────────────────────
    st.markdown('<p class="section-header">🔧 Controls</p>', unsafe_allow_html=True)
    if st.button("🗑️ Clear Conversation History", use_container_width=True):
        st.session_state.query_history = []
        try:
            requests.post(f"{API_BASE}/clear-history", timeout=5)
        except Exception:
            pass
        st.success("History cleared!")

    st.markdown("---")

    # ── ML Delay Prediction ─────────────────────────────────────────────────────
    st.markdown('<p class="section-header">🤖 Delay Prediction</p>', unsafe_allow_html=True)

    if health and not health.get("model_trained"):
        st.warning("Model not trained. Click below to train.")
        if st.button("🚀 Train Delay Model", use_container_width=True):
            with st.spinner("Training XGBoost model... (~30s)"):
                result = train_model()
            if "error" in result:
                st.error(f"Training failed: {result['error']}")
            else:
                st.success(f"✅ Trained! Accuracy: {result.get('accuracy', 0):.2%}")
                st.rerun()

    with st.form("delay_form"):
        st.markdown("**Predict Delivery Delay**")
        dist = st.number_input("Distance (km)", 10.0, 5000.0, 200.0, 10.0)
        days_sched = st.number_input("Scheduled Days", 1, 30, 3)
        traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High", "Very High"])
        ship_mode = st.selectbox("Shipping Mode", ["Same Day", "First Class", "Second Class", "Standard Class"])
        exp_years = st.slider("Driver Experience (yrs)", 1, 20, 3)
        rating = st.slider("Driver Rating", 1.0, 5.0, 3.5, 0.1)
        hour = st.slider("Hour of Day", 0, 23, 12)

        predict_btn = st.form_submit_button("🔮 Predict Risk", use_container_width=True)

    if predict_btn:
        if health and health.get("model_trained"):
            features = {
                "distance_km": dist,
                "days_scheduled": int(days_sched),
                "traffic_level": traffic,
                "shipping_mode": ship_mode,
                "experience_years": int(exp_years),
                "rating": float(rating),
                "hour_of_day": int(hour),
                "day_of_week": 1,
            }
            with st.spinner("Predicting..."):
                result = predict_delay(features)
            if "error" in result:
                st.error(result["error"])
            else:
                prob = result.get("delay_chance_pct", 0)
                label = result.get("risk_label", "")
                css_cls = "risk-high" if "HIGH" in label else "risk-low"
                st.markdown(
                    f'<div class="{css_cls}">{label}<br>'
                    f'<span style="font-size:0.9rem;font-weight:400">{prob}% probability</span></div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("Train the model first using the button above.")


# ─────────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────────

# Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🚚 Logistics AI Analytics Agent</div>
    <div class="hero-sub">Ask questions in natural language · Auto SQL generation · Real-time insights</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Stats row (top metrics)
if health and health.get("database") == "connected":
    try:
        schema_data_cached = get_schema()
        table_count = len(schema_data_cached.get("schema_dict", {}))
    except Exception:
        table_count = 7

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">DataCo</div>
            <div class="metric-label">Supply Chain Dataset</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{table_count}</div>
            <div class="metric-label">Database Tables</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">Groq</div>
            <div class="metric-label">Llama 3.3 70B</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">XGBoost</div>
            <div class="metric-label">Delay Predictor</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# ── Query Input ──────────────────────────────────────────────────────────────────
if "selected_question" in st.session_state:
    st.session_state["main_question_input"] = st.session_state.pop("selected_question")

question = st.text_input(
    "💬 Ask a logistics question",
    placeholder="e.g. Which delivery routes have the highest delay rate?",
    key="main_question_input",
)

col_ask, col_new = st.columns([3, 1])
with col_ask:
    ask_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")
with col_new:
    new_btn = st.button("🆕 New Conversation", use_container_width=True)

if new_btn:
    st.session_state.query_history = []
    try:
        requests.post(f"{API_BASE}/clear-history", timeout=5)
    except Exception:
        pass
    st.session_state.last_result = None
    st.rerun()

# ── Run Query ────────────────────────────────────────────────────────────────────
if (ask_btn or st.session_state.get("trigger_query")) and question.strip():
    st.session_state["trigger_query"] = False  # Reset trigger
    if not health:
        st.error("❌ Cannot connect to backend. Make sure `uvicorn main:app --reload` is running.")
    else:
        with st.spinner("🤖 Thinking..."):
            start_t = time.time()
            result = run_query(question)
            elapsed = round(time.time() - start_t, 2)

        if result.get("error") and not result.get("sql"):
            st.markdown(f'<div class="error-block">❌ {result["error"]}</div>', unsafe_allow_html=True)
        else:
            st.session_state.last_result = result
            st.session_state.query_history.append({
                "question": question,
                "sql": result.get("sql", ""),
                "row_count": result.get("row_count", 0),
                "elapsed": elapsed,
            })

# ── Display Result ────────────────────────────────────────────────────────────────
result = st.session_state.last_result
if result:
    steps = result.get("steps", [])
    is_multi = len(steps) > 1

    # Step tags
    if is_multi:
        st.markdown('<p class="section-header">🗺️ Query Plan</p>', unsafe_allow_html=True)
        for i, s in enumerate(steps, 1):
            st.markdown(f'<span class="step-tag">Step {i}: {s}</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Layout: SQL + Insight (top row)
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<p class="section-header">🔧 Generated SQL</p>', unsafe_allow_html=True)
        sql_display = result.get("sql", "No SQL generated")
        st.markdown(f'<div class="sql-block">{sql_display}</div>', unsafe_allow_html=True)
        row_count = result.get("row_count", 0)
        st.caption(f"↳ {row_count} rows returned")

    with col_right:
        st.markdown('<p class="section-header">💡 Business Insight</p>', unsafe_allow_html=True)
        insight = result.get("insight", "No insight available.")
        if result.get("error"):
            st.markdown(f'<div class="error-block">{result["error"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="insight-block">{insight}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Chart
    chart_json = result.get("chart_json")
    if chart_json:
        st.markdown('<p class="section-header">📊 Visualization</p>', unsafe_allow_html=True)
        try:
            fig = pio.from_json(chart_json)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Chart could not be rendered: {e}")

    # Results Table
    results_data = result.get("results", [])
    if results_data:
        st.markdown('<p class="section-header">📋 Query Results</p>', unsafe_allow_html=True)
        df = pd.DataFrame(results_data)
        st.dataframe(
            df,
            use_container_width=True,
            height=min(400, 50 + 35 * len(df)),
        )

        # Download
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name="query_results.csv",
            mime="text/csv",
        )

# ── Query History ─────────────────────────────────────────────────────────────────
if st.session_state.query_history:
    st.markdown("---")
    st.markdown('<p class="section-header">🕘 Query History</p>', unsafe_allow_html=True)
    for i, h in enumerate(reversed(st.session_state.query_history[-8:])):
        with st.expander(f"Q{len(st.session_state.query_history) - i}: {h['question'][:80]}..."):
            st.markdown(f'<div class="sql-block">{h["sql"]}</div>', unsafe_allow_html=True)
            st.caption(f"↳ {h['row_count']} rows · {h['elapsed']}s")
            if st.button("Re-run this query", key=f"rerun_{i}"):
                st.session_state["selected_question"] = h["question"]
                st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#334155; font-size:0.8rem; border-top:1px solid #1e3a5f; padding-top:16px;">
    🚚 Logistics AI Analytics Agent · Powered by Groq (Llama 3.3 70B) + LangChain + PostgreSQL
</div>
""", unsafe_allow_html=True)

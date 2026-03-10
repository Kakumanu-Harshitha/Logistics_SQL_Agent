"""
app.py
Logistics Insight AI - Refined UI based on reference design.
Features: Teal accents, side-by-side status, and delay prediction.
"""
import streamlit as st
import requests
import pandas as pd
import plotly.io as pio
import plotly.express as px
import json
import time
import uuid

# ─── Configuration ──────────────────────────────────────────────────────────────
API_BASE = "https://logistics-sql-agent.onrender.com"

st.set_page_config(
    page_title="Logistics Insight AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS for Reference Design ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-main: #020617;
    --bg-card: #0F172A;
    --accent: #14b8a6; /* Teal accent */
    --accent-glow: rgba(20, 184, 166, 0.3);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --border: #1e293b;
    --success: #10b981;
}

* { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: var(--bg-main);
    color: var(--text-primary);
}

/* Header Styling */
.header-container {
    padding: 1rem 0;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.header-icon {
    background-color: #134e4a;
    padding: 10px;
    border-radius: 10px;
}

.header-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
}

.header-subtitle {
    font-size: 0.875rem;
    color: var(--text-secondary);
    margin: 0;
}

/* Metric Card Styling */
.metric-card {
    background-color: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
}

.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
}

.metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.metric-trend {
    font-size: 0.875rem;
    font-weight: 500;
}

.trend-up { color: #4ade80; }
.trend-down { color: #f87171; }

.metric-icon {
    position: absolute;
    top: 1.5rem;
    right: 1.5rem;
    color: var(--accent);
    opacity: 0.8;
}

/* Query Input Styling */
.query-container {
    background-color: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.75rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
}

.stTextInput > div > div > input {
    background-color: #0f172a !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
    border-radius: 8px !important;
}

.stButton > button {
    background-color: #0f2a28 !important; /* Dark teal background */
    color: #2dd4bf !important;
    border: 1px solid #115e59 !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background-color: #115e59 !important;
    box-shadow: 0 0 10px var(--accent-glow) !important;
}

/* Prediction Button (Cyan Gradient) */
.predict-btn > div > button {
    background: linear-gradient(90deg, #2dd4bf, #06b6d4) !important;
    color: #020617 !important;
    border: none !important;
    width: 100% !important;
    padding: 0.75rem !important;
    font-size: 1rem !important;
}

/* Panels Styling */
.side-panel {
    background-color: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
}

.panel-header {
    font-size: 0.875rem;
    font-weight: 700;
    color: var(--text-secondary);
    text-transform: uppercase;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Status Item */
.status-item {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1.5rem;
}

.status-dot {
    width: 8px;
    height: 8px;
    background-color: var(--success);
    border-radius: 50%;
    margin-top: 6px;
    box-shadow: 0 0 8px var(--success);
}

.status-label {
    font-weight: 600;
    font-size: 0.9375rem;
}

.status-subtext {
    font-size: 0.75rem;
    color: var(--text-secondary);
}

.status-badge {
    font-size: 0.625rem;
    font-weight: 700;
    color: var(--success);
    letter-spacing: 0.05em;
}

/* Suggestions Box */
.suggestion-pill > div > button {
    background-color: #020617 !important;
    border: 1px solid var(--border) !important;
    border-radius: 30px !important;
    padding: 4px 12px !important;
    font-size: 0.8rem !important;
    color: #2dd4bf !important;
    width: auto !important;
    margin: 0 !important;
}

.suggestion-pill > div > button:hover {
    border-color: var(--accent) !important;
}

/* SQL Block */
.stCodeBlock { background: #020617 !important; border: 1px solid var(--border) !important; }

</style>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────────────────────────
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "current_query" not in st.session_state:
    st.session_state.current_query = ""

# ─── Helper Functions ────────────────────────────────────────────────────────────
def check_api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def run_query(question: str):
    payload = {"question": question, "session_id": st.session_state.session_id}
    try:
        r = requests.post(f"{API_BASE}/query", json=payload, timeout=60)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}

def get_metrics():
    try:
        r = requests.get(f"{API_BASE}/dashboard-metrics", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def predict_delay_api(params: dict):
    try:
        r = requests.post(f"{API_BASE}/predict-delay", json=params, timeout=10)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}

# ─── Main UI Context ─────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="header-container">
    <div class="header-icon">🧠</div>
    <div>
        <p class="header-title">Logistics Insight AI</p>
        <p class="header-subtitle">Analytics & Delay Prediction</p>
    </div>
    <div style="margin-left: auto; display: flex; align-items: center; gap: 8px;">
        <div style="width: 8px; height: 8px; border-radius: 50%; background: #4ade80;"></div>
        <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">LIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Metric Row
metrics = get_metrics() or {
    "total_orders": 180519,
    "avg_delivery_days": 4.1,
    "late_delivery_rate": 39.7,
    "revenue_str": "$13.4M",
    "trends": {"orders": "+12.3%", "delivery": "-0.3 days", "late_rate": "+2.1%", "revenue": "+8.7%"}
}

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">📦</div>
        <p class="metric-label">Total Orders</p>
        <p class="metric-value">{metrics['total_orders']:,}</p>
        <p class="metric-trend trend-up">↑ {metrics['trends']['orders']} <span style="color:#64748b; font-weight:400; font-size:0.75rem">vs last month</span></p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">🕒</div>
        <p class="metric-label">Avg Delivery</p>
        <p class="metric-value">{metrics['avg_delivery_days']} days</p>
        <p class="metric-trend trend-up">↓ {metrics['trends']['delivery']} <span style="color:#64748b; font-weight:400; font-size:0.75rem">improved</span></p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">⚠️</div>
        <p class="metric-label">Late Delivery Rate</p>
        <p class="metric-value">{metrics['late_delivery_rate']}%</p>
        <p class="metric-trend trend-down">↑ {metrics['trends']['late_rate']} <span style="color:#64748b; font-weight:400; font-size:0.75rem">from baseline</span></p>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">📈</div>
        <p class="metric-label">Revenue</p>
        <p class="metric-value">{metrics['revenue_str']}</p>
        <p class="metric-trend trend-up">↑ {metrics['trends']['revenue']} <span style="color:#64748b; font-weight:400; font-size:0.75rem">YoY growth</span></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Query Section ───
q_col, a_col = st.columns([6, 1])
with q_col:
    query_text = st.text_input(
        "Query", 
        value=st.session_state.current_query,
        label_visibility="collapsed", 
        placeholder="Ask a logistics question in plain English...",
    )
with a_col:
    analyze_btn = st.button("🔍 Analyze", use_container_width=True)

# Suggestion Pills
suggestions = [
    "Which cities have the highest delivery delays?",
    "What product category generates the most revenue?",
    "Which shipping mode has the highest late delivery rate?",
    "What is the average delivery time by region?"
]

s_cols = st.columns(len(suggestions))
for idx, s in enumerate(suggestions):
    st.markdown('<div class="suggestion-pill">', unsafe_allow_html=True)
    if s_cols[idx].button(s, key=f"sug_{idx}"):
        st.session_state.current_query = s
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Results & Panels row ───
main_col, side_col = st.columns([2, 1], gap="large")

with main_col:
    if st.session_state.last_result:
        res = st.session_state.last_result
        if res.get("error"):
            st.error(res["error"])
        else:
            # Vertical sequence as requested
            st.markdown("---")
            
            # 1. SQL Query
            st.markdown('<div class="section-title">� SQL Query</div>', unsafe_allow_html=True)
            st.code(res.get("sql", "-- No SQL produced"), language="sql")
            st.markdown("<br>", unsafe_allow_html=True)

            # 2. Results
            st.markdown('<div class="section-title">📋 Results</div>', unsafe_allow_html=True)
            df = pd.DataFrame(res.get("results", []))
            if not df.empty:
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No data found.")
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 3. Visualization
            st.markdown('<div class="section-title">📊 Visualization</div>', unsafe_allow_html=True)
            chart_json = res.get("chart_json")
            if chart_json:
                try:
                    fig = pio.from_json(chart_json)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.caption(f"Could not render chart: {e}")
            elif not df.empty:
                # Fallback chart if no chart_json
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0 and len(df.columns) > 1:
                    fig = px.bar(df, x=df.columns[0], y=numeric_cols[0], template="plotly_dark")
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("No suitable data for visualization.")
            else:
                st.caption("No data to visualize.")
            st.markdown("<br>", unsafe_allow_html=True)

            # 4. Business Insights
            st.markdown('<div class="section-title">💡 Business Insights</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="css-card">
                <p style="color: var(--text-primary); line-height: 1.6; margin: 0;">
                    {res.get('insight', 'No insights available.')}
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Initial Placeholder
        st.markdown("""
        <div style="text-align:center; padding: 100px 0; border: 1px dashed var(--border); border-radius: 12px;">
            <div style="font-size: 3rem; opacity: 0.2; margin-bottom: 1rem;">🧠</div>
            <p style="color: var(--text-secondary)">Ask a logistics question above or click a suggestion to get started.</p>
        </div>
        """, unsafe_allow_html=True)

with side_col:
    # System Status Panel
    st.markdown("""
    <div class="side-panel">
        <div class="panel-header">📈 System Status</div>
        <div class="status-item">
            <div>
                <p class="status-label">AI Model</p>
                <p class="status-subtext">Gemini 3 Flash</p>
            </div>
            <div style="text-align: right;">
                <div class="status-dot"></div>
                <p class="status-badge">ONLINE</p>
            </div>
        </div>
        <div class="status-item">
            <div>
                <p class="status-label">Database</p>
                <p class="status-subtext">PostgreSQL - 8 tables</p>
            </div>
            <div style="text-align: right;">
                <div class="status-dot"></div>
                <p class="status-badge">ONLINE</p>
            </div>
        </div>
        <div class="status-item">
            <div>
                <p class="status-label">ML Model</p>
                <p class="status-subtext">XGBoost Delay Predictor</p>
            </div>
            <div style="text-align: right;">
                <div class="status-dot"></div>
                <p class="status-badge">ONLINE</p>
            </div>
        </div>
        <div class="status-item">
            <div>
                <p class="status-label">Dataset</p>
                <p class="status-subtext">180,519 records loaded</p>
            </div>
            <div style="text-align: right;">
                <div class="status-dot"></div>
                <p class="status-badge">ONLINE</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Delay Prediction Panel
    with st.container(border=True):
        st.markdown('<p style="font-size: 0.875rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">🚛 Delay Prediction</p>', unsafe_allow_html=True)
        
        pk_col1, pk_col2 = st.columns(2)
        with pk_col1:
            dist = st.number_input("DISTANCE (KM)", value=450, min_value=1)
            traffic = st.selectbox("TRAFFIC LEVEL", ["Low", "Medium", "High", "Very High"])
            exp = st.number_input("DRIVER EXP (YRS)", value=3, min_value=0)
            hour = st.slider("HOUR OF DAY", 0, 23, 12)
        with pk_col2:
            days = st.number_input("SCHEDULED DAYS", value=5, min_value=1)
            mode = st.selectbox("SHIPPING MODE", ["Standard Class", "First Class", "Second Class", "Same Day"])
            rating = st.number_input("DRIVER RATING", value=4.2, min_value=1.0, max_value=5.0)
            dow = st.selectbox("DAY OF WEEK", options=[0,1,2,3,4,5,6], format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
        
        st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
        if st.button("Predict Delay Risk"):
            params = {
                "distance_km": float(dist),
                "days_scheduled": int(days),
                "traffic_level": traffic,
                "shipping_mode": mode,
                "experience_years": int(exp),
                "rating": float(rating),
                "hour_of_day": int(hour),
                "day_of_week": int(dow)
            }
            with st.spinner("Analyzing..."):
                res = predict_delay_api(params)
                if "error" in res:
                    st.error(f"Prediction failed: {res['error']}")
                else:
                    risk = res.get("risk_label", "UNKNOWN")
                    prob = res.get("delay_chance_pct", 0)
                    color = "#f87171" if risk == "HIGH RISK" else "#4ade80"
                    
                    st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.5); padding: 15px; border-radius: 8px; border: 1px solid var(--border); margin-top: 10px;">
                        <p style="font-size: 0.75rem; color: #94a3b8; margin: 0;">PREDICTED RISK</p>
                        <p style="font-size: 1.25rem; font-weight: 700; color: {color}; margin: 5px 0;">{risk}</p>
                        <p style="font-size: 0.875rem; color: #f8fafc; margin: 0;">Probability: {prob}%</p>
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ─── Execution ───
if analyze_btn and query_text:
    with st.spinner("Processing..."):
        result = run_query(query_text)
        st.session_state.last_result = result
        st.rerun()

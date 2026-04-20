from __future__ import annotations

import io
import os
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Rossmann Sales Forecast",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# GLOBAL CSS  (dark theme)
# ──────────────────────────────────────────────
def inject_css() -> None:
    dark = True
    page_bg    = "#0F172A" if dark else "#F4F6FB"
    card_bg    = "#1E293B" if dark else "#FFFFFF"
    text_col   = "#F1F5F9" if dark else "#1C1C2E"
    muted_col  = "#94A3B8" if dark else "#4B5563"
    border_col = "#334155" if dark else "#E5E7EB"
    sidebar_bg = "#0B1120" if dark else "#1C1C2E"
    input_bg   = "#1E293B" if dark else "#FFFFFF"
    st.markdown(f"""
    <style>
        /* ── Palette ── */
        :root {{
            --red:      #E3001B;
            --red-soft: #FF4D5E;
            --dark:     {sidebar_bg};
            --card-bg:  {card_bg};
            --page-bg:  {page_bg};
            --text:     {text_col};
            --muted:    {muted_col};
            --border:   {border_col};
        }}

        /* ── Page background ── */
        .stApp, .stApp > div {{ background: {page_bg} !important; }}

        /* ── Global text ── */
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
        .stApp p, .stApp li, .stApp td, .stApp th,
        section[data-testid="stMain"] span,
        section[data-testid="stMain"] label,
        section[data-testid="stMain"] .stMarkdown *,
        section[data-testid="stMain"] .stNumberInput input,
        section[data-testid="stMain"] .stTextInput input,
        section[data-testid="stMain"] .stDateInput input {{
            color: {text_col} !important;
        }}
        section[data-testid="stMain"] input,
        section[data-testid="stMain"] textarea {{
            background-color: {input_bg} !important;
            color: {text_col} !important;
        }}

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab"] {{ color: {text_col} !important; }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            color: #E3001B !important;
            border-bottom-color: #E3001B !important;
        }}

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {{ background: {sidebar_bg} !important; }}
        [data-testid="stSidebar"] * {{ color: #F0F0F0 !important; }}
        [data-testid="stSidebar"] .stRadio label {{ font-size: 0.95rem; }}

        /* ── Top header bar ── */
        .rossmann-header {{
            background: linear-gradient(135deg, #E3001B 0%, #A80013 100%);
            padding: 1.4rem 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        .rossmann-header * {{ color: #fff !important; }}
        .rossmann-header h1 {{ margin: 0; font-size: 1.8rem; font-weight: 700; }}
        .rossmann-header p  {{ margin: 0; opacity: .85; font-size: 0.95rem; }}

        /* ── Metric card ── */
        .metric-card {{
            background: {card_bg};
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            box-shadow: 0 1px 4px rgba(0,0,0,.15);
            border-left: 4px solid #E3001B;
        }}
        .metric-card .label {{ color: {muted_col} !important; font-size: .8rem; text-transform: uppercase; letter-spacing: .05em; }}
        .metric-card .value {{ color: {text_col} !important; font-size: 1.7rem; font-weight: 700; margin-top: .2rem; }}

        /* ── Section title ── */
        .section-title {{
            font-size: 1.1rem; font-weight: 700;
            color: {text_col} !important; margin-bottom: .8rem;
            border-bottom: 2px solid #E3001B;
            padding-bottom: .35rem;
        }}

        /* ── Status pill ── */
        .pill-ok  {{ background:#D1FAE5; color:#065F46 !important; padding:.25rem .7rem; border-radius:999px; font-size:.8rem; font-weight:600; }}
        .pill-err {{ background:#FEE2E2; color:#991B1B !important; padding:.25rem .7rem; border-radius:999px; font-size:.8rem; font-weight:600; }}

        /* ── Result highlight ── */
        .result-box {{
            background: linear-gradient(135deg, #FFF1F2 0%, #FFF 100%);
            border: 2px solid #FF4D5E;
            border-radius: 12px;
            padding: 1.4rem;
            text-align: center;
        }}
        .result-box .big {{ font-size: 2.5rem; font-weight: 800; color: #E3001B !important; }}
        .result-box .sub {{ color: {muted_col} !important; font-size: .9rem; margin-top: .3rem; }}

        /* ── Buttons ── */
        div[data-testid="stButton"] > button {{
            background: #E3001B; color: #fff !important;
            border: none; border-radius: 8px;
            padding: .55rem 1.6rem; font-weight: 600; transition: background .2s;
        }}
        div[data-testid="stButton"] > button:hover {{ background: #FF4D5E; }}
        .stDownloadButton > button {{
            background: {sidebar_bg} !important;
            color: #fff !important; border-radius: 8px !important;
        }}
    </style>
    """, unsafe_allow_html=True)


inject_css()

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
DAY_LABELS = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
HOLIDAY_OPTS = {"None (0)": "0", "Public holiday (a)": "a", "Easter (b)": "b", "Christmas (c)": "c"}

# Test data date range – restrict all predictions to this window
TEST_DATE_MIN = date(2015, 8, 1)
TEST_DATE_MAX = date(2015, 9, 17)


@st.cache_data
def load_holiday_lookup() -> dict:
    """Load Open/StateHoliday/SchoolHoliday from the processed test CSV keyed by (Store, Date).
    Falls back to date-only lookup for StateHoliday/SchoolHoliday when Store not found.
    Returns empty dict when the file has not been pulled yet."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "test.csv")
    try:
        df = pd.read_csv(data_path, usecols=["Store", "Date", "Open", "StateHoliday", "SchoolHoliday"])
        df["StateHoliday"] = df["StateHoliday"].astype(str).replace({"nan": "0"})
        # Normalize Date to YYYY-MM-DD string regardless of CSV format
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        df["Store"] = df["Store"].astype(int)
        # Open có NaN trong Kaggle test (store đóng cửa không có doanh số) → fill 0
        df["Open"] = df["Open"].fillna(1).astype(int)
        df["SchoolHoliday"] = df["SchoolHoliday"].fillna(0).astype(int)
        # Primary lookup: (Store, Date) → full info including Open
        # Use itertuples to ensure Python native int keys (avoid numpy.int64 mismatch)
        store_date = {
            (int(row.Store), str(row.Date)): {
                "Open": int(row.Open),
                "StateHoliday": str(row.StateHoliday),
                "SchoolHoliday": int(row.SchoolHoliday),
            }
            for row in df.itertuples(index=False)
        }
        # Fallback lookup: Date → holiday info (date-level, deduplicated)
        date_only = df.drop_duplicates("Date").set_index("Date")[["StateHoliday", "SchoolHoliday"]]
        return {"store_date": store_date, "date_only": date_only.to_dict("index")}
    except Exception:
        return {}


def api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def call_predict(records: list[dict]) -> list[float] | None:
    try:
        r = requests.post(f"{API_URL}/predict", json={"records": records}, timeout=30)
        r.raise_for_status()
        return r.json()["predictions"]
    except requests.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None


def rossmann_chart_theme() -> dict:
    tc   = "#F1F5F9"
    grid = "#334155"
    axis_style = dict(
        color=tc,
        tickfont=dict(color=tc, size=12),
        title_font=dict(color=tc, size=13),
        gridcolor=grid,
        zerolinecolor=grid,
    )
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=tc),
        title_font=dict(color=tc, size=15),
        xaxis=axis_style,
        yaxis=axis_style,
        legend=dict(font=dict(color=tc)),
        margin=dict(l=20, r=20, t=40, b=20),
    )


def chart_text_color() -> str:
    return "#F1F5F9"


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Rossmann MLOps")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠  Dashboard", "🔮  Single Prediction", "📦  Batch Prediction"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    healthy = api_health()
    if healthy:
        st.markdown('<span class="pill-ok">● API Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill-err">● API Offline</span>', unsafe_allow_html=True)
    st.caption(f"Endpoint: `{API_URL}`")

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class="rossmann-header">
        <div>
            <h1>🛒 Rossmann Sales Forecast</h1>
            <p>Predict daily store sales using historical retail data & machine learning</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════
if page == "🏠  Dashboard":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="metric-card"><div class="label">Model</div>'
            '<div class="value">RandomForest</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><div class="label">API Status</div>'
            f'<div class="value">{"✅ Online" if healthy else "❌ Offline"}</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><div class="label">Dataset</div>'
            '<div class="value">Rossmann Retail</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column layout ──
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown('<div class="section-title">📋 Project Overview</div>', unsafe_allow_html=True)
        st.markdown(
            """
            This application demonstrates an **end-to-end MLOps pipeline** for forecasting
            daily sales across **1,115 Rossmann drugstores** in Germany.

            | Component | Technology |
            |-----------|------------|
            | Model | RandomForestRegressor |
            | Serving | FastAPI + Uvicorn |
            | UI | Streamlit |
            | Tracking | MLflow |
            | Versioning | DVC |
            | Container | Docker Compose |
            """
        )

    with right:
        st.markdown('<div class="section-title">📅 Quick Demo</div>', unsafe_allow_html=True)
        st.markdown("Generate a sample weekly forecast for any store.")

        demo_store = st.number_input("Store ID", min_value=1, max_value=1115, value=1, key="demo_store")
        demo_start = st.date_input(
            "Week starting",
            value=date(2015, 9, 11),
            min_value=TEST_DATE_MIN,
            max_value=date(2015, 9, 11),  # cap so the 7-day window stays within test period
            help=f"Demo window is 7 days; capped so all days fall within {TEST_DATE_MIN} – {TEST_DATE_MAX}.",
            key="demo_date",
        )

        if st.button("Run Demo Forecast"):
            if not healthy:
                st.warning("API is offline. Start the API service first.")
            else:
                demo_lookup = load_holiday_lookup()
                demo_store_key = int(demo_store)
                demo_sdlookup = demo_lookup.get("store_date", {})
                demo_datelookup = demo_lookup.get("date_only", {})

                records = []
                for i in range(7):
                    d = demo_start + timedelta(days=i)
                    d_str = d.isoformat()
                    dow = d.isoweekday()
                    s_info = demo_sdlookup.get((demo_store_key, d_str), {})
                    d_info = demo_datelookup.get(d_str, {})
                    if s_info:
                        open_ = int(s_info.get("Open", 1))
                        sh = str(s_info.get("StateHoliday", "0"))
                        sch = int(s_info.get("SchoolHoliday", 0))
                    else:
                        open_ = 1
                        sh = str(d_info.get("StateHoliday", "0"))
                        sch = int(d_info.get("SchoolHoliday", 0))
                    records.append({
                        "Store": demo_store_key,
                        "DayOfWeek": dow,
                        "Date": d_str,
                        "Open": open_,
                        "Promo": 1 if i < 5 else 0,
                        "StateHoliday": sh,
                        "SchoolHoliday": sch,
                    })

                with st.spinner("Forecasting…"):
                    preds = call_predict(records)
                if preds:
                    # Stores closed (Open=0) → force 0 in chart
                    preds = [p if records[i]["Open"] == 1 else 0.0 for i, p in enumerate(preds)]
                    days = [(demo_start + timedelta(days=i)).isoformat() for i in range(7)]
                    df_demo = pd.DataFrame({"Date": days, "Predicted Sales (€)": preds})
                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=df_demo["Date"],
                            y=df_demo["Predicted Sales (€)"],
                            marker_color="#E3001B",
                            text=[f"€{v:,.0f}" for v in preds],
                            textposition="outside",
                            textfont=dict(color=chart_text_color(), size=12),
                        )
                    )
                    fig.update_layout(
                        title=f"Store {demo_store} – Weekly Forecast",
                        **rossmann_chart_theme(),
                    )
                    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE: SINGLE PREDICTION
# ══════════════════════════════════════════════
elif page == "🔮  Single Prediction":
    st.markdown('<div class="section-title">🔮 Single Store Prediction</div>', unsafe_allow_html=True)

    holiday_lookup = load_holiday_lookup()

    st.info(
        "ℹ️ **Open** status is looked up from actual test data per store and date (some stores open on Sundays). "
        "**State Holiday** and **School Holiday** are also looked up from historical calendar data. "
        f"Predictions are restricted to the test period: **{TEST_DATE_MIN}** – **{TEST_DATE_MAX}**."
    )

    with st.form("single_form"):
        c1, c2 = st.columns(2)
        with c1:
            store = st.number_input("Store ID", min_value=1, max_value=1115, value=1)
            pred_date = st.date_input(
                "Date",
                value=date(2015, 9, 14),
                min_value=TEST_DATE_MIN,
                max_value=TEST_DATE_MAX,
                help=f"Only dates within the test period ({TEST_DATE_MIN} – {TEST_DATE_MAX}) are valid.",
            )
        with c2:
            promo = st.selectbox("Promo", [1, 0], format_func=lambda x: "Active" if x else "Inactive")

        submitted = st.form_submit_button("Predict Sales →", use_container_width=True)

    if submitted:
        if not healthy:
            st.warning("API is offline. Please start the API service.")
        else:
            day_of_week = pred_date.isoweekday()
            date_str = pred_date.isoformat()

            # Look up per-store Open status from actual test data
            # Cast store to int: st.number_input returns float (e.g. 1.0) which won't match int key
            store_key = int(store)
            store_date_lookup = holiday_lookup.get("store_date", {})
            date_only_lookup = holiday_lookup.get("date_only", {})
            store_info = store_date_lookup.get((store_key, date_str), {})
            date_info = date_only_lookup.get(date_str, {})

            if store_info:
                open_ = int(store_info.get("Open", 0 if day_of_week == 7 else 1))
                state_holiday = str(store_info.get("StateHoliday", "0"))
                school_holiday = int(store_info.get("SchoolHoliday", 0))
            else:
                # Fallback khi store không có trong test data: mặc định mở cửa
                open_ = 1
                state_holiday = str(date_info.get("StateHoliday", "0"))
                school_holiday = int(date_info.get("SchoolHoliday", 0))

            record = {
                "Store": store_key,
                "DayOfWeek": day_of_week,
                "Date": date_str,
                "Open": open_,
                "Promo": promo,
                "StateHoliday": state_holiday,
                "SchoolHoliday": school_holiday,
            }

            sh_label = {
                "0": "None", "a": "Public Holiday", "b": "Easter", "c": "Christmas"
            }.get(state_holiday, state_holiday)

            st.markdown("<br>", unsafe_allow_html=True)
            res_col, info_col = st.columns([1, 1.3], gap="large")

            if open_ == 0:
                # Store đóng cửa → không gọi API, trả sales = 0
                with res_col:
                    st.markdown(
                        f"""
                        <div class="result-box">
                            <div class="sub">Predicted Sales</div>
                            <div class="big">€ 0.00</div>
                            <div class="sub">Store {store_key} · {pred_date.strftime('%A, %d %b %Y')} — Closed</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with info_col:
                    st.info("🔒 This store is **closed** on this date according to the test data. Sales = 0.")
            else:
                # Store mở cửa → gọi API dự báo
                with st.spinner("Running prediction…"):
                    preds = call_predict([record])
                if preds:
                    with res_col:
                        st.markdown(
                            f"""
                            <div class="result-box">
                                <div class="sub">Predicted Sales</div>
                                <div class="big">€ {preds[0]:,.2f}</div>
                                <div class="sub">Store {store_key} · {pred_date.strftime('%A, %d %b %Y')}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with info_col:
                        st.markdown('<div class="section-title">Input Summary</div>', unsafe_allow_html=True)
                        summary = pd.DataFrame(
                            {
                                "Field": ["Store", "Date", "Day of Week", "Open (auto)", "Promo", "State Holiday (auto)", "School Holiday (auto)"],
                                "Value": [
                                    store_key,
                                    pred_date.strftime("%d %b %Y"),
                                    DAY_LABELS[day_of_week],
                                    "Yes",
                                    "Active" if promo else "Inactive",
                                    sh_label,
                                    "Yes" if school_holiday else "No",
                                ],
                            }
                        )
                        st.dataframe(summary, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE: BATCH PREDICTION
# ══════════════════════════════════════════════
elif page == "📦  Batch Prediction":
    st.markdown('<div class="section-title">📦 Batch Prediction</div>', unsafe_allow_html=True)

    tab_upload, tab_generate = st.tabs(["📁 Upload CSV", "🗓️ Generate Date Range"])

    # ── Tab 1: Upload ──
    with tab_upload:
        st.markdown(
            "Upload a CSV with columns: **Store, DayOfWeek, Date, Open, Promo, StateHoliday, SchoolHoliday**"
        )
        uploaded = st.file_uploader("Choose CSV file", type=["csv"])

        if uploaded:
            df_up = pd.read_csv(uploaded)
            st.markdown(f"**{len(df_up)} rows loaded** — preview:")
            st.dataframe(df_up.head(10), use_container_width=True)

            if st.button("Run Batch Prediction", key="upload_btn"):
                if not healthy:
                    st.warning("API is offline.")
                else:
                    records = df_up.to_dict(orient="records")
                    with st.spinner(f"Predicting {len(records)} records…"):
                        preds = call_predict(records)
                    if preds:
                        df_up["Predicted_Sales"] = preds
                        st.success(f"Done! {len(preds)} predictions generated.")

                        fig = px.line(
                            df_up,
                            x="Date",
                            y="Predicted_Sales",
                            color="Store" if "Store" in df_up.columns else None,
                            title="Batch Forecast Results",
                            labels={"Predicted_Sales": "Sales (€)"},
                            color_discrete_sequence=px.colors.qualitative.Set1,
                        )
                        fig.update_layout(**rossmann_chart_theme())
                        st.plotly_chart(fig, use_container_width=True)

                        csv_out = df_up.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇ Download Results CSV",
                            data=csv_out,
                            file_name="rossmann_predictions.csv",
                            mime="text/csv",
                        )

    # ── Tab 2: Generate ──
    with tab_generate:
        with st.form("gen_form"):
            g1, g2, g3 = st.columns(3)
            with g1:
                g_stores = st.text_input("Store IDs (comma-separated)", value="1,2,3")
                g_start = st.date_input(
                    "Start Date",
                    value=date(2015, 9, 1),
                    min_value=TEST_DATE_MIN,
                    max_value=TEST_DATE_MAX,
                )
            with g2:
                g_end = st.date_input(
                    "End Date",
                    value=date(2015, 9, 14),
                    min_value=TEST_DATE_MIN,
                    max_value=TEST_DATE_MAX,
                )
                g_open = st.selectbox("Open", [1, 0], format_func=lambda x: "Yes" if x else "No")
            with g3:
                g_promo = st.selectbox("Promo", [1, 0], format_func=lambda x: "Active" if x else "Inactive")
                g_school = st.selectbox("School Holiday", [0, 1], format_func=lambda x: "Yes" if x else "No")

            gen_submitted = st.form_submit_button("Generate & Predict →", use_container_width=True)

        if gen_submitted:
            if not healthy:
                st.warning("API is offline.")
            else:
                try:
                    store_ids = [int(s.strip()) for s in g_stores.split(",") if s.strip()]
                except ValueError:
                    st.error("Invalid store IDs. Enter comma-separated integers.")
                    st.stop()

                gen_lookup = load_holiday_lookup()
                gen_sdlookup = gen_lookup.get("store_date", {})
                gen_datelookup = gen_lookup.get("date_only", {})

                date_range = pd.date_range(g_start, g_end)
                records = []
                for sid in store_ids:
                    for d in date_range:
                        d_str = d.date().isoformat()
                        dow = d.isoweekday()
                        s_info = gen_sdlookup.get((sid, d_str), {})
                        d_info = gen_datelookup.get(d_str, {})
                        if s_info:
                            open_ = int(s_info.get("Open", 1))
                            sh = str(s_info.get("StateHoliday", "0"))
                            sch = int(s_info.get("SchoolHoliday", 0))
                        else:
                            open_ = g_open
                            sh = str(d_info.get("StateHoliday", "0"))
                            sch = g_school
                        records.append({
                            "Store": sid,
                            "DayOfWeek": dow,
                            "Date": d_str,
                            "Open": open_,
                            "Promo": g_promo,
                            "StateHoliday": sh,
                            "SchoolHoliday": sch,
                        })

                if len(records) > 500:
                    st.warning(f"{len(records)} records — this may take a moment.")

                with st.spinner(f"Predicting {len(records)} records…"):
                    preds = call_predict(records)

                if preds:
                    # Force 0 for closed stores
                    preds = [p if records[i]["Open"] == 1 else 0.0 for i, p in enumerate(preds)]
                    df_gen = pd.DataFrame(records)
                    df_gen["Predicted_Sales"] = preds

                    st.success(f"Done! {len(preds)} predictions.")

                    # ── Summary metrics ──
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(
                            f'<div class="metric-card"><div class="label">Total Forecast</div>'
                            f'<div class="value">€ {sum(preds):,.0f}</div></div>',
                            unsafe_allow_html=True,
                        )
                    with m2:
                        st.markdown(
                            f'<div class="metric-card"><div class="label">Avg Daily Sales</div>'
                            f'<div class="value">€ {sum(preds)/len(preds):,.0f}</div></div>',
                            unsafe_allow_html=True,
                        )
                    with m3:
                        st.markdown(
                            f'<div class="metric-card"><div class="label">Peak Day</div>'
                            f'<div class="value">€ {max(preds):,.0f}</div></div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown("<br>", unsafe_allow_html=True)

                    # ── Chart ──
                    fig = px.line(
                        df_gen,
                        x="Date",
                        y="Predicted_Sales",
                        color="Store",
                        title="Forecast by Store & Date",
                        labels={"Predicted_Sales": "Sales (€)", "Store": "Store"},
                        color_discrete_sequence=["#E3001B", "#FF6B6B", "#FFAA00", "#2D9CDB", "#27AE60", "#8E44AD"],
                    )
                    fig.update_traces(mode="lines+markers", marker_size=5)
                    fig.update_layout(**rossmann_chart_theme())
                    st.plotly_chart(fig, use_container_width=True)

                    # ── Bar by store ──
                    df_by_store = df_gen.groupby("Store")["Predicted_Sales"].sum().reset_index()
                    fig2 = px.bar(
                        df_by_store,
                        x="Store",
                        y="Predicted_Sales",
                        title="Total Forecast by Store",
                        labels={"Predicted_Sales": "Total Sales (€)"},
                        color="Predicted_Sales",
                        color_continuous_scale=["#FFD6D9", "#E3001B"],
                    )
                    fig2.update_layout(**rossmann_chart_theme())
                    st.plotly_chart(fig2, use_container_width=True)

                    csv_out = df_gen.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇ Download Results CSV",
                        data=csv_out,
                        file_name="rossmann_batch_predictions.csv",
                        mime="text/csv",
                    )

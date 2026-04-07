import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="SOC Dashboard", layout="wide")

st.title("🛡️ Real-Time SOC Dashboard")
st.success("🟢 System Status: Monitoring Active")

# ===============================
# LOAD MODEL
# ===============================
try:
    model_data = joblib.load("full_pipeline.pkl")
    model = model_data["model"]
    scaler = model_data["scaler"]
    features = model_data["features"]
except:
    st.error("❌ Model not found. Run training first.")
    st.stop()

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

threshold_percent = st.sidebar.slider("Alert Threshold (%)", 90, 99, 95)

auto_refresh = st.sidebar.checkbox("🔄 Enable Live Mode")

selected_user = st.sidebar.text_input("🔍 Track User ID (optional)")

# ===============================
# LOAD DATA
# ===============================
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validate features
    missing = [f for f in features if f not in df.columns]
    if missing:
        st.error(f"❌ Missing columns: {missing}")
        st.stop()

    # ===============================
    # SESSION STATE (NO BLINK)
    # ===============================
    if "live_data" not in st.session_state:
        st.session_state.live_data = df.sample(min(100, len(df))).copy()

    df_live = st.session_state.live_data.copy()

    # ===============================
    # SIMULATE LIVE DATA
    # ===============================
    for col in features:
        df_live[col] += np.random.normal(0, 0.02, len(df_live))

    st.session_state.live_data = df_live

    # ===============================
    # MODEL PREDICTION
    # ===============================
    X = df_live[features]
    X_scaled = scaler.transform(X)

    scores = model.predict_proba(X_scaled)[:,1]
    df_live["Risk Score"] = scores

    # ===============================
    # ALERT THRESHOLD
    # ===============================
    threshold = np.percentile(scores, threshold_percent)
    alerts = df_live[df_live["Risk Score"] > threshold]

    # ===============================
    # SEVERITY
    # ===============================
    def get_severity(score):
        if score > 0.9:
            return "CRITICAL"
        elif score > 0.75:
            return "HIGH"
        elif score > 0.5:
            return "MEDIUM"
        else:
            return "LOW"

    df_live["Severity"] = df_live["Risk Score"].apply(get_severity)

    # ===============================
    # METRICS
    # ===============================
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🚨 Alerts", len(alerts))
    c2.metric("📊 Avg Risk", round(scores.mean(), 3))
    c3.metric("🔥 Max Risk", round(scores.max(), 3))
    c4.metric("👥 Users", len(df_live))

    # ===============================
    # TABS
    # ===============================
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚨 Alerts",
        "📊 Analytics",
        "👤 User Tracking",
        "📋 Data"
    ])

    # ===============================
    # ALERTS TAB
    # ===============================
    with tab1:

        st.subheader("🚨 Live Alerts")

        if len(alerts) > 0:
            st.error("⚠️ Critical Activity Detected")

            alerts_sorted = alerts.sort_values("Risk Score", ascending=False)

            st.dataframe(alerts_sorted.head(10))

            st.download_button(
                "⬇️ Download Alerts",
                alerts_sorted.to_csv(index=False),
                file_name="alerts.csv"
            )

        else:
            st.success("✅ System Normal")

    # ===============================
    # ANALYTICS TAB
    # ===============================
    with tab2:

        st.subheader("📊 Risk Distribution")
        st.bar_chart(df_live["Risk Score"])

        st.subheader("📊 Severity Breakdown")
        st.write(df_live["Severity"].value_counts())

        st.subheader("📈 Statistics")
        st.write(df_live["Risk Score"].describe())

    # ===============================
    # USER TRACKING TAB
    # ===============================
    with tab3:

        st.subheader("👤 User Tracking")

        if "user" in df_live.columns:

            users = df_live["user"].unique()

            user_select = st.selectbox("Select User", users)

            user_data = df_live[df_live["user"] == user_select]

            st.write("### User Activity")
            st.dataframe(user_data)

            st.write("### Risk Trend")
            st.line_chart(user_data["Risk Score"])

        else:
            st.warning("⚠️ 'user' column not found")

    # ===============================
    # DATA TAB
    # ===============================
    with tab4:

        st.subheader("🔥 Top Risky Users")
        st.dataframe(
            df_live.sort_values("Risk Score", ascending=False).head(20)
        )

        st.subheader("📋 Full Dataset")
        st.dataframe(df_live)

    # ===============================
    # LIVE MODE LOOP (NO FULL REFRESH)
    # ===============================
    if auto_refresh:
        time.sleep(2)
        st.rerun()

else:
    st.info("📂 Upload ui_input.csv to start")
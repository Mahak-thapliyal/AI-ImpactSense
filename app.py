import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="Earthquake Alert Predictor",
    page_icon="🌍",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------ Load CSS & HTML ------------------
def load_css():
    css_path = os.path.join(BASE_DIR, "static", "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_html():
    html_path = os.path.join(BASE_DIR, "templates", "index.html")
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            st.markdown(f.read(), unsafe_allow_html=True)

load_css()
load_html()

# ------------------ Load Model ------------------
try:
    model = joblib.load("gradient_boosting_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file not found!")
    st.stop()

# ------------------ Sidebar ------------------
st.sidebar.header("⚙️ Earthquake Parameters")

scenario = st.sidebar.selectbox(
    "📌 Select Scenario",
    ["Custom", "Mild", "Moderate", "Severe"]
)

if scenario == "Mild":
    magnitude, depth, cdi, mmi, sig = 3.2, 50, 2.0, 2.5, 80
elif scenario == "Moderate":
    magnitude, depth, cdi, mmi, sig = 5.6, 30, 4.5, 5.0, 300
elif scenario == "Severe":
    magnitude, depth, cdi, mmi, sig = 8.2, 15, 8.5, 9.0, 850
else:
    magnitude = st.sidebar.slider("🌍 Magnitude", 0.0, 10.0, 5.5)
    depth = st.sidebar.number_input("⛏️ Depth (km)", 0, 700, 10)
    cdi = st.sidebar.slider("📈 CDI", 0.0, 10.0, 3.0)
    mmi = st.sidebar.slider("📊 MMI", 0.0, 10.0, 4.0)
    sig = st.sidebar.number_input("⚠️ Significance (SIG)", -100, 1000, 150)

dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown(
        "<style>body{background-color:#0f172a;color:white;}</style>",
        unsafe_allow_html=True
    )

auto_predict = st.sidebar.toggle("🔄 Auto Predict", value=True)

# ------------------ Prediction ------------------
if auto_predict or st.button("🔮 Predict Alert"):

    with st.spinner("🔍 Predicting alert level..."):

        input_df = pd.DataFrame(
            [[magnitude, depth, cdi, mmi, sig]],
            columns=['magnitude', 'depth', 'cdi', 'mmi', 'sig']
        )

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        alert_map = {0: "green", 1: "yellow", 2: "red"}
        alert_label = alert_map.get(prediction, "unknown")

        # ------------------ Alert Display ------------------
        st.markdown("## 🚨 Predicted Alert Level")

        if alert_label == "green":
            st.success("🟢 **GREEN – Low Risk**")
        elif alert_label == "yellow":
            st.warning("🟡 **YELLOW – Moderate Risk**")
        elif alert_label == "red":
            st.error("🔴 **RED – High Risk**")
        else:
            st.info("⚠️ Unknown Alert")

        # ------------------ Alert Meter ------------------
        st.markdown("### 📊 Alert Meter")

        alert_score = {"green": 30, "yellow": 65, "red": 90}
        st.progress(alert_score.get(alert_label, 0))

        # ------------------ Confidence Score ------------------
        confidence = max(probabilities) * 100
        st.metric("📌 Prediction Confidence", f"{confidence:.2f}%")

        # ------------------ Feature Visualization ------------------
        st.markdown("### 📈 Input Feature Overview")

        chart_df = pd.DataFrame({
            "Feature": ["Magnitude", "Depth", "CDI", "MMI", "SIG"],
            "Value": [magnitude, depth, cdi, mmi, sig]
        }).set_index("Feature")

        st.bar_chart(chart_df)

        # ------------------ Explanation ------------------
        with st.expander("📖 How this alert was predicted"):
            st.write(f"""
            - **Magnitude:** {magnitude}
            - **Depth:** {depth} km
            - **CDI:** {cdi}
            - **MMI:** {mmi}
            - **Significance:** {sig}

            The Gradient Boosting model evaluates seismic intensity,
            depth, and impact indicators to classify the alert level.
            """)

# ------------------ Footer ------------------
st.markdown("---")
st.markdown(
    "📌 *This application uses a trained Gradient Boosting model for earthquake alert prediction.*"
)




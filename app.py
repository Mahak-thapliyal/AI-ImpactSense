import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="Earthquake Alert Predictor",
    page_icon="🌍",
    layout="wide"
)

# ------------------ Load Model & Preprocessing ------------------
try:
    model = joblib.load("gradient_boosting_model.pkl")
    scaler = joblib.load("scaler3.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except FileNotFoundError:
    st.error("❌ Model / Scaler / Encoder file not found!")
    st.stop()

# ------------------ CSS for better UI ------------------
st.markdown("""
<style>
.card {
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
    color: white;
    font-weight: bold;
}
.green {background-color: #28a745;}
.yellow {background-color: #ffc107; color: black;}
.orange {background-color: #fd7e14;}
.red {background-color: #dc3545;}
h2 {color: #0f172a;}
</style>
""", unsafe_allow_html=True)

# ------------------ Page Title ------------------
st.title("🌍 Earthquake Alert Predictor")
st.write("Enter earthquake parameters below to predict the alert level:")

# ------------------ Inputs & Prediction in Columns ------------------
col1, col2 = st.columns([1, 1])

with col1:
    magnitude = st.number_input("🌍 Magnitude", 0.0, 10.0, 5.5)
    depth = st.number_input("⛏️ Depth (km)", 0.0, 700.0, 10.0)
    cdi = st.number_input("📈 CDI", 0.0, 10.0, 3.0)
    mmi = st.number_input("📊 MMI", 0.0, 10.0, 4.0)
    sig = st.number_input("⚠️ Significance (SIG)", -1000.0, 1000.0, 150.0)

with col2:
    if st.button("🔮 Predict Alert"):

        input_df = pd.DataFrame([[magnitude, depth, cdi, mmi, sig]],
                                columns=['magnitude', 'depth', 'cdi', 'mmi', 'sig'])
        input_scaled = scaler.transform(input_df.values)

        pred_encoded = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]

        alert_label = label_encoder.inverse_transform([pred_encoded])[0]

        # ------------------ Alert Card ------------------
        alert_colors = {"green": "green", "yellow": "yellow", "orange": "orange", "red": "red"}
        st.markdown(f"<div class='card {alert_colors.get(alert_label, '')}'>Predicted Alert: {alert_label.upper()}</div>", unsafe_allow_html=True)

        # ------------------ Alert Meter ------------------
        alert_score = {"green": 30, "yellow": 50, "orange": 75, "red": 90}
        st.progress(alert_score.get(alert_label, 0))

        # ------------------ Confidence ------------------
        confidence = max(probabilities) * 100
        st.metric("📌 Prediction Confidence", f"{confidence:.2f}%")

        # ------------------ Probability Table ------------------
        prob_df = pd.DataFrame({
            "Alert Level": label_encoder.inverse_transform(range(len(probabilities))),
            "Probability (%)": (probabilities * 100).round(2)
        }).set_index("Alert Level")
        st.markdown("### 📊 Probability for Each Alert Level")
        st.bar_chart(prob_df)

        # ------------------ Feature Visualization ------------------
        chart_df = pd.DataFrame({
            "Feature": ["Magnitude", "Depth", "CDI", "MMI", "SIG"],
            "Value": [magnitude, depth, cdi, mmi, sig]
        }).set_index("Feature")
        st.markdown("### 🔹 Input Feature Overview")
        st.bar_chart(chart_df)

        # ------------------ Explanation ------------------
        with st.expander("📖 How this alert was predicted"):
            st.write(f"""
            - **Magnitude:** {magnitude}
            - **Depth:** {depth} km
            - **CDI:** {cdi}
            - **MMI:** {mmi}
            - **Significance:** {sig}

            The Gradient Boosting model evaluates earthquake intensity,
            depth, and impact indicators after standard scaling,
            and classifies the alert into four levels: green, yellow, orange, or red.
            """)

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("📌 *This app uses a trained Gradient Boosting model with enhanced UI and four alert levels.*")


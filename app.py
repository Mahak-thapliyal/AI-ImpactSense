import streamlit as st
import pandas as pd
import joblib

# ================== Page Configuration ==================
st.set_page_config(
    page_title="Earthquake Alert Predictor",
    page_icon="🌍",
    layout="wide"
)

# ================== Load Model ==================
try:
    model = joblib.load("gradient_boosting_model.pkl")
    scaler = joblib.load("scaler3.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except FileNotFoundError:
    st.error("❌ Model / Scaler / Encoder file not found!")
    st.stop()

# ================== SOFT NEON CSS ==================
st.markdown("""
<style>
/* Background */
body {
    background-color: #0b0f14;
}

/* TITLE (SOFT NEON) */
.main-title {
    font-size: 44px;
    font-weight: 900;
    color: #7cffb2;        /* Soft neon mint */
    text-align: center;
    text-shadow: 0 0 4px rgba(124,255,178,0.4);
}

/* SUBTITLE */
.subtext {
    color: #d1d5db;
    font-size: 17px;
    text-align: center;
}

/* SECTION HEADERS */
.section {
    font-size: 22px;
    font-weight: 700;
    color: #93c5fd;        /* Soft sky neon */
    text-shadow: 0 0 3px rgba(147,197,253,0.35);
    margin-top: 20px;
}

/* ALERT CARD */
.card {
    padding: 24px;
    border-radius: 16px;
    margin-top: 26px;
    font-size: 28px;
    font-weight: 800;
    text-align: center;
    background: #111827;
    border: 1px solid #1f2937;
}

/* ALERT COLORS (SUBTLE GLOW) */
.green  { color: #86efac; border-left: 6px solid #22c55e; }
.yellow { color: #fde68a; border-left: 6px solid #eab308; }
.orange { color: #fdba74; border-left: 6px solid #f97316; }
.red    { color: #fca5a5; border-left: 6px solid #ef4444; }
</style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
st.markdown("<div class='main-title'>🌍 EARTHQUAKE ALERT PREDICTOR</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'></div>", unsafe_allow_html=True)
st.divider()

# ================== SIDEBAR INPUTS ==================
st.sidebar.header("Input Parameters")

magnitude = st.sidebar.text_input("Magnitude", "5.5")
depth = st.sidebar.text_input("Depth (km)", "10")
cdi = st.sidebar.text_input("CDI", "3")
mmi = st.sidebar.text_input("MMI", "4")
sig = st.sidebar.text_input("Significance (SIG)", "150")

predict_btn = st.sidebar.button("Predict Alert")

# ================== PREDICTION ==================
if predict_btn:
    try:
        magnitude = float(magnitude)
        depth = float(depth)
        cdi = float(cdi)
        mmi = float(mmi)
        sig = float(sig)

        input_df = pd.DataFrame(
            [[magnitude, depth, cdi, mmi, sig]],
            columns=["magnitude", "depth", "cdi", "mmi", "sig"]
        )

        input_scaled = scaler.transform(input_df)

        pred_encoded = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        alert_label = label_encoder.inverse_transform([pred_encoded])[0]

        # ALERT CARD
        st.markdown(
            f"<div class='card {alert_label}'>🚨 ALERT LEVEL: {alert_label.upper()}</div>",
            unsafe_allow_html=True
        )

        # CONFIDENCE
        confidence = max(probabilities) * 100
        st.markdown("<div class='section'>Prediction Confidence</div>", unsafe_allow_html=True)
        st.progress(int(confidence))
        st.write(f"Confidence: {confidence:.2f}%")

        st.divider()

        # INPUT SUMMARY
        st.markdown("<div class='section'>Input Summary</div>", unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Magnitude", magnitude)
        col2.metric("Depth (km)", depth)
        col3.metric("CDI", cdi)
        col4.metric("MMI", mmi)
        col5.metric("SIG", sig)

        # PROBABILITY CHART
        prob_df = pd.DataFrame({
            "Alert Level": label_encoder.inverse_transform(range(len(probabilities))),
            "Probability (%)": (probabilities * 100).round(2)
        }).set_index("Alert Level")

        st.markdown("<div class='section'>Alert Probability Distribution</div>", unsafe_allow_html=True)
        st.bar_chart(prob_df)

        # FEATURE CHART
        feature_df = pd.DataFrame({
            "Feature": ["Magnitude", "Depth", "CDI", "MMI", "SIG"],
            "Value": [magnitude, depth, cdi, mmi, sig]
        }).set_index("Feature")

        st.markdown("<div class='section'>Input Feature Overview</div>", unsafe_allow_html=True)
        st.bar_chart(feature_df)

 

    except ValueError:
        st.error("Please enter valid numeric values.")

# ================== FOOTER ==================
st.divider()
st.markdown("Earthquake Alert Predictor")




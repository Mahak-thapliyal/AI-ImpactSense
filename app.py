import streamlit as st
import pandas as pd
import joblib

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="Earthquake Alert Predictor",
    layout="centered"
)
def load_css():
    with open("static/style.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_html():
    with open("templates/index.html", encoding="utf-8") as f:
        st.markdown(f.read(), unsafe_allow_html=True)


load_css()
load_html()

# ------------------ Load Model ------------------
try:
    rf_model = joblib.load(".venv/gradient_boosting_model.pkl")
except FileNotFoundError:
    st.error("❌ gradient_boosting_model.pkl file not found! Please place it in the same folder.")
    st.stop()

# ------------------ Input Section ------------------
st.subheader("📊 Enter Earthquake Parameters")

magnitude = st.slider("Magnitude", 0.0, 10.0, 5.5)
depth = st.number_input("Depth (km)", 0, 700, 10)
cdi = st.slider("CDI", 0.0, 10.0, 3.0)
mmi = st.slider("MMI", 0.0, 10.0, 4.0)
sig = st.number_input("Significance (SIG)", -100, 1000, 150)

# ------------------ Prediction ------------------
if st.button("🔮 Predict Alert"):
    input_df = pd.DataFrame(
        [[magnitude, depth, cdi, mmi, sig]],
        columns=['magnitude', 'depth', 'cdi', 'mmi', 'sig']
    )

    prediction = rf_model.predict(input_df)[0]

    # Map prediction to alert label
    alert_map = {
        0: "green",
        1: "yellow",
        2: "red"
    }

    alert_label = alert_map.get(prediction, "unknown")

    # Display result with color
    if alert_label == "green":
        st.success("🟢 Predicted Alert Level: **GREEN**")
    elif alert_label == "yellow":
        st.warning("🟡 Predicted Alert Level: **YELLOW**")
    elif alert_label == "red":
        st.error("🔴 Predicted Alert Level: **RED**")
    else:
        st.info("⚠️ Prediction result unknown")

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("📌 *This app uses a trained Gradient Boosting model for earthquake alert prediction.*")

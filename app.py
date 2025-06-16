import streamlit as st
import joblib
import numpy as np

# Load pre-trained scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# Page config
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

# Sidebar branding
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4149/4149687.png", width=100)  # new icon
    st.markdown("## Customer Churn Predictor")
    st.markdown("Built by Diya Koyani\n\nPredict if a customer is likely to churn based on 4 key attributes.")

# Main Title
st.title("🔍 Churn Prediction Engine")

st.markdown("---")
st.write("📥 **Input the customer's details below to predict if they are likely to churn.**")

# Input fields
age = st.slider("📅 Age", min_value=10, max_value=100, value=30, help="Customer's age")
gender = st.radio("👤 Gender", ["Male", "Female"])
tenure = st.slider("📈 Tenure (months)", min_value=0, max_value=130, value=10, help="How long has the customer been with the company?")
monthlycharge = st.slider("💸 Monthly Charges", min_value=30, max_value=150, value=50, help="Monthly fee charged to the customer")

# Button
if st.button("Predict Churn"):
    gender_encoded = 1 if gender == "Female" else 0
    user_input = np.array([[age, gender_encoded, tenure, monthlycharge]])
    scaled_input = scaler.transform(user_input)
    
    prediction = model.predict(scaled_input)[0]

    st.success("✅ Prediction complete!")

    # Show result
    if prediction == 1:
        st.markdown(f"### 🚨 This customer is **likely to churn**.")
        st.image("https://cdn-icons-png.flaticon.com/512/7541/7541900.png", width=80)  # new warning icon
    else:
        st.markdown(f"### 🎉 This customer is **not likely to churn**.")
        st.image("https://cdn-icons-png.flaticon.com/512/148/148767.png", width=80)  # new success icon
    
    st.balloons()

else:
    st.info("Please enter the values and press **Predict Churn** to get started.")

# Footer
st.markdown("---")
st.caption("📌 Note: Predictions are based on logistic classification using Age, Gender, Tenure, and Monthly Charges.")

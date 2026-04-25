import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from db import *

# ===============================
# INIT DB
# ===============================
init_db()

# ===============================
# LOAD MODEL
# ===============================
model_path = os.path.join(os.path.dirname(__file__), "heart_model.pkl")
model = joblib.load(model_path)

# ===============================
# SESSION STATE
# ===============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ===============================
# LOGIN / SIGNUP
# ===============================
if not st.session_state.logged_in:
    st.title("🔐 Login / Signup")

    choice = st.selectbox("Choose", ["Login", "Signup"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if choice == "Signup":
        if st.button("Create Account"):
            if signup_user(username, password):
                st.success("Account created!")
            else:
                st.error("User already exists")

    if choice == "Login":
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged In!")
            else:
                st.error("Invalid credentials")

# ===============================
# MAIN APP
# ===============================
if st.session_state.logged_in:

    st.sidebar.title(f"Welcome {st.session_state.username}")
    page = st.sidebar.radio("Navigate", ["Prediction", "Dashboard", "Logout"])

    if page == "Logout":
        st.session_state.logged_in = False
        st.experimental_rerun()

    # ===============================
    # PREDICTION PAGE
    # ===============================
    if page == "Prediction":
        st.title("❤️ Heart Disease Prediction")

        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 20, 100, 50)
            sex = st.selectbox("Sex", [0,1])
            cp = st.selectbox("Chest Pain", [0,1,2,3])
            trestbps = st.slider("BP", 80, 200, 120)
            chol = st.slider("Cholesterol", 100, 400, 200)

        with col2:
            thalach = st.slider("Max HR", 60, 220, 150)
            oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
            slope = st.selectbox("Slope", [0,1,2])
            ca = st.selectbox("Vessels", [0,1,2,3])
            thal = st.selectbox("Thal", [0,1,2,3])

        input_data = pd.DataFrame({
            'age':[age],'sex':[sex],'cp':[cp],
            'trestbps':[trestbps],'chol':[chol],
            'fbs':[0],'restecg':[0],'thalach':[thalach],
            'exang':[0],'oldpeak':[oldpeak],
            'slope':[slope],'ca':[ca],'thal':[thal]
        })

        # Feature Engineering
        input_data['age_chol'] = input_data['age'] * input_data['chol']
        input_data['bp_chol'] = input_data['trestbps'] * input_data['chol']
        input_data['chol_norm'] = input_data['chol'] / 400
        input_data['bp_norm'] = input_data['trestbps'] / 200
        input_data['risk_score'] = (
            input_data['age']*0.2 +
            input_data['chol']*0.2 +
            input_data['trestbps']*0.2 +
            input_data['thalach']*0.2 +
            input_data['oldpeak']*0.2
        )

        if st.button("Predict"):
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            if prediction == 1:
                st.error(f"⚠️ High Risk ({probability:.2f})")
            else:
                st.success(f"✅ Low Risk ({probability:.2f})")

            save_prediction(
                st.session_state.username,
                age, chol, trestbps,
                prediction, probability
            )

            st.bar_chart(input_data.T)

    # ===============================
    # DASHBOARD
    # ===============================
    if page == "Dashboard":
        st.title("📊 Dashboard")

        data = get_user_predictions(st.session_state.username)

        if len(data) == 0:
            st.warning("No predictions yet")
        else:
            df = pd.DataFrame(data)

            st.dataframe(df)

            st.subheader("Prediction Distribution")
            st.bar_chart(df['prediction'].value_counts())

            st.subheader("Cholesterol vs BP")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='chol', y='bp', hue='prediction', ax=ax)
            st.pyplot(fig)
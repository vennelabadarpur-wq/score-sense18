import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


st.set_page_config(page_title="score-sense", page_icon="🎓", layout="centered")


st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #667eea, #764ba2);
        color: white;
    }
    .stButton>button {
        background-color: #ff4b2b;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    .stNumberInput>div>div>input {
        background-color: #f0f2f6;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)


np.random.seed(42)
rows = 2000

social = np.random.randint(0, 5, rows)
study = np.random.uniform(1, 10, rows).round(1)
sleep = np.random.uniform(6, 9, rows).round(1)
attendance = np.random.randint(70, 101, rows)


gpa = 6 + (0.25*study) + (0.015*attendance) + (0.1*sleep) - (0.2*social)
gpa = np.clip(gpa, 6, 9.6).round(2)

df = pd.DataFrame({
    "Social": social,
    "Study": study,
    "Sleep": sleep,
    "Attendance": attendance,
    "GPA": gpa
})

X = df[["Social","Study","Sleep","Attendance"]]
y = df["GPA"]

model = LinearRegression()
model.fit(X,y)


st.title("🎓 score-sense (6 – 9.6 Scale)")
st.write("### Enter Student Details 👇")

social_input = st.number_input("📱 Social Media Hours", 0.0, 10.0, 1.0)
study_input = st.number_input("📚 Study Hours", 0.0, 12.0, 5.0)
sleep_input = st.number_input("😴 Sleep Hours", 0.0, 12.0, 7.0)
attendance_input = st.number_input("🏫 Attendance %", 0, 100, 80)

if st.button("🚀 Predict GPA"):
    data = np.array([[social_input, study_input, sleep_input, attendance_input]])
    prediction = model.predict(data)[0]
    prediction = round(prediction,2)
    
    st.success(f"🎯 Predicted GPA: {prediction}")
    
    # GPA Color Indicator
    if prediction >= 9:
        st.balloons()
        st.markdown("### 🌟 Excellent Performance!")
    elif prediction >= 8:
        st.markdown("### 👍 Very Good Performance!")
    elif prediction >= 7:
        st.markdown("### 🙂 Good Performance!")
    else:
        st.markdown("### ⚠️ Need Improvement!")
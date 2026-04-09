import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------- DATA ----------------
df = pd.read_csv("student_data.csv")
df.columns = df.columns.str.strip()

X = df[['study_hours', 'attendance', 'previous_marks']]
y = df['final_marks']

model = LinearRegression()
model.fit(X, y)

# ---------------- PAGE ----------------
st.set_page_config(page_title="Student AI Predictor", layout="wide")

st.title("📊 Student Performance AI Dashboard")

# ---------------- INPUTS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    study_hours = st.number_input("Study Hours", 0.0, 24.0, 5.0)

with col2:
    attendance = st.number_input("Attendance (%)", 0.0, 100.0, 75.0)

with col3:
    previous_marks = st.number_input("Previous Marks", 0.0, 100.0, 60.0)

predict = st.button("🚀 Predict")

# ---------------- CHARTS ----------------
st.subheader("📈 Data Insights")

colA, colB = st.columns(2)

with colA:
    fig1 = plt.figure()
    plt.scatter(df['study_hours'], df['final_marks'])
    plt.xlabel("Study Hours")
    plt.ylabel("Final Marks")
    plt.title("Study Hours vs Marks")
    st.pyplot(fig1)

with colB:
    fig2 = plt.figure()
    plt.scatter(df['attendance'], df['final_marks'])
    plt.xlabel("Attendance")
    plt.ylabel("Final Marks")
    plt.title("Attendance vs Marks")
    st.pyplot(fig2)

# ---------------- PREDICTION ----------------
if predict:
    input_data = pd.DataFrame([[study_hours, attendance, previous_marks]],
                              columns=['study_hours', 'attendance', 'previous_marks'])

    prediction = model.predict(input_data)[0]

    st.success(f"🎯 Predicted Final Marks: {prediction:.2f}")

    # live point chart
    fig3 = plt.figure()
    plt.scatter(df['study_hours'], df['final_marks'], label="Data")
    plt.scatter(study_hours, prediction, color='red', s=100, label="Your Input")
    plt.xlabel("Study Hours")
    plt.ylabel("Final Marks")
    plt.legend()
    plt.title("Your Prediction vs Data")
    st.pyplot(fig3)

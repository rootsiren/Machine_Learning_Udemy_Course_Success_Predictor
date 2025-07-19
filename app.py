import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load the pipeline
model = joblib.load("model.pkl")

st.title("📈 Course Popularity Predictor")

# State control
if "show_inputs" not in st.session_state:
    st.session_state.show_inputs = False

if st.sidebar.button("Predict now"):
    st.session_state.show_inputs = True

# Show form only after button click
if st.session_state.show_inputs:
    st.subheader("Enter course details")

    course_title = st.text_input("Enter Course Title")
    paid = st.radio("Is this course paid?", ["True", "False"])
    price = st.slider("Price", 0, 200, 20)
    num_reviews = st.number_input("Number of reviews", min_value=0)
    num_lectures = st.number_input("Number of lectures", min_value=0)
    level = st.radio("Course Level", ['All Levels', 'Intermediate Level', 'Beginner Level', 'Expert Level'])
    content_duration = st.slider("Content Duration (hrs)", 0.0, 80.0, 1.0)
    subject = st.selectbox("Subject", ['Business Finance', 'Graphic Design', 'Musical Instruments', 'Web Development'])
    published_date = st.date_input("Published Date")

    course_age_days = (datetime.today().date() - published_date).days

    # Create input DataFrame
    input_df = pd.DataFrame([{
        'course_title': course_title,
        'is_paid': 1 if paid == "True" else 0,
        'price': price,
        'num_reviews': num_reviews,
        'num_lectures': num_lectures,
        'level': level,
        'content_duration': content_duration,
        'subject': subject,
        'course_age_days': course_age_days
    }])

    if st.button("Predict"):
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.success(f"Predicted: {'Popular' if prediction else 'Not Popular'}")
        st.info(f"Probability: {round(probability * 100, 2)}%")

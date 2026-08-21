
import streamlit as st
import joblib
import re
import string

# Load trained model and vectorizer
model = joblib.load("classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# Clean resume text
def clean_resume(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Predict resume
def predict_resume(resume_text):

    cleaned_text = clean_resume(resume_text)

    resume_vector = vectorizer.transform([cleaned_text])

    prediction = model.predict(resume_vector)[0]

    fit_index = list(model.classes_).index("Fit")

    fit_score = (
        model.predict_proba(resume_vector)[0][fit_index] * 100
    )

    return prediction, fit_score


# App title
st.title("🤖 AI Resume Screening System")

st.write(
    "Upload a resume and the trained machine learning model "
    "will evaluate its suitability."
)

# Resume input
resume_text = st.text_area(
    "Paste Resume Text",
    height=300
)

# Screen button
if st.button("Screen Resume"):

    if resume_text.strip():

        prediction, fit_score = predict_resume(resume_text)

        st.subheader("Screening Result")

        st.write(f"**Prediction:** {prediction}")
        st.write(f"**Fit Score:** {fit_score:.2f}%")

    else:
        st.warning("Please enter a resume.")

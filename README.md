
# AI Resume Screening System

An AI-powered resume screening system that uses machine learning and skill matching to evaluate resumes against predefined job categories and required skills.

## Features

- Resume text screening
- TF-IDF text representation
- Logistic Regression classification
- Random Oversampling for class imbalance
- Fit probability score
- Required-skill matching
- Matched and missing skills identification

## Machine Learning Pipeline

Resume
↓
Text Cleaning
↓
TF-IDF Vectorization
↓
Logistic Regression
↓
Fit / Not Fit Prediction
↓
Skill Matching
↓
Screening Result

## Model

The model was trained using:

- TF-IDF Vectorizer
- Logistic Regression
- Random Oversampling

The "Fit" category consists of:

- Information Technology
- Engineering
- Digital Media

## Model Performance

The final test evaluation achieved approximately 92% accuracy.

The model achieved approximately 84% recall for the Fit class.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Joblib
- Streamlit

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt

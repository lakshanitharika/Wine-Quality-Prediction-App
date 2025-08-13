import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix

# Load model and data
model = joblib.load("model.pkl")
data = pd.read_csv("data/winequality-red.csv")

# Convert target to binary
data['quality_label'] = (data['quality'] >= 7).astype(int)

st.set_page_config(page_title="Wine Quality Prediction App", layout="centered")

# Sidebar
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to:", ["📊 Data Exploration", "📈 Visualizations", "🍷 Make Prediction", "📋 Model Performance"])

# Data Exploration
if option == "📊 Data Exploration":
    st.title("Wine Dataset Overview")
    st.write(data.head())
    st.write(f"Shape: {data.shape}")
    st.write(data.describe())

# Visualization
elif option == "📈 Visualizations":
    st.title("Data Visualizations")
    st.subheader("Quality Distribution")
    st.bar_chart(data['quality'].value_counts())

    st.subheader("Alcohol vs Quality")
    fig = px.scatter(data, x="alcohol", y="quality", color="quality_label")
    st.plotly_chart(fig)

    st.subheader("Correlation Heatmap")
    fig2, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig2)

# Prediction
elif option == "🍷 Make Prediction":
    st.title("Wine Quality Prediction")
    st.markdown("Enter wine chemical composition values:")

    fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 8.0)
    volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.5)
    citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
    residual_sugar = st.slider("Residual Sugar", 0.5, 15.0, 2.0)
    chlorides = st.slider("Chlorides", 0.01, 0.2, 0.05)
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1.0, 72.0, 15.0)
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6.0, 289.0, 50.0)
    density = st.slider("Density", 0.9900, 1.0040, 0.9960)
    pH = st.slider("pH", 2.5, 4.5, 3.3)
    sulphates = st.slider("Sulphates", 0.3, 2.0, 0.6)
    alcohol = st.slider("Alcohol", 8.0, 15.0, 10.0)

    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                            density, pH, sulphates, alcohol]])
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    st.success("Good Quality Wine 🍷" if prediction == 1 else "Low Quality Wine 🍷")
    st.info(f"Prediction Confidence: {proba[prediction]*100:.2f}%")

# Model Performance
elif option == "📋 Model Performance":
    st.title("Model Performance")
    X = data.drop(['quality', 'quality_label'], axis=1)
    y_true = data['quality_label']
    y_pred = model.predict(X)

    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig3, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig3)

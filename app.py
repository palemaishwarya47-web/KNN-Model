import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Page Configuration
st.set_page_config(
    page_title="KNN Iris Classifier",
    page_icon="🌸",
    layout="centered"
)

# Title
st.title("🌸 Iris Flower Classification using KNN")
st.write(
    "This machine learning app predicts the species of an Iris flower "
    "based on sepal and petal measurements using the K-Nearest Neighbors (KNN) algorithm."
)

# Load Dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
knn_model = KNeighborsClassifier(
    n_neighbors=5,
    metric='minkowski',
    p=2
)
knn_model.fit(X_train, y_train)

# Model Accuracy
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Sidebar
st.sidebar.header("Input Flower Measurements")

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)", 4.0, 8.0, 5.8
)
sepal_width = st.sidebar.slider(
    "Sepal Width (cm)", 2.0, 4.5, 3.0
)
petal_length = st.sidebar.slider(
    "Petal Length (cm)", 1.0, 7.0, 4.0
)
petal_width = st.sidebar.slider(
    "Petal Width (cm)", 0.1, 2.5, 1.2
)

# Input Data
input_data = np.array([
    [
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]
])

# Prediction
prediction = knn_model.predict(input_data)
predicted_species = target_names[prediction[0]]

# Display Results
st.subheader("📊 Input Measurements")
st.write(f"**Sepal Length:** {sepal_length} cm")
st.write(f"**Sepal Width:** {sepal_width} cm")
st.write(f"**Petal Length:** {petal_length} cm")
st.write(f"**Petal Width:** {petal_width} cm")

st.subheader("🌿 Prediction Result")
st.success(f"Predicted Iris Species: **{predicted_species.capitalize()}**")

st.subheader("📈 Model Performance")
st.info(f"KNN Model Accuracy: **{accuracy:.2f}**")

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Machine Learning Project - KNN Iris Classification")
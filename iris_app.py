import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load iris dataset
iris = load_iris()
X = iris.data
Y = iris.target

# Set up classifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Create Streamlit app
st.title("Iris Flower Prediction App")
st.header("Enter the Sepal and Petal Details")

# Set up input fields
sepal_length = st.slider("Sepal length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider("Sepal width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider("Petal length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider("Petal width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Define prediction button
if st.button("Predict"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = clf.predict(input_data)
    st.write(f"The predicted type of iris flower is {iris.target_names[prediction[0]]}")

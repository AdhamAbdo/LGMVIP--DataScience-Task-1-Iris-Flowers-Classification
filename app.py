import streamlit as st
import pickle
import numpy as np

# Load the pickled model
with open('SVM.pickle', 'rb') as f:
    model = pickle.load(f)

# Define a function to predict the class of a new data point
def predict(X_new):
    X_new = X_new.reshape(1, -1)
    prediction = model.predict(X_new)
    return prediction

# Create a Streamlit app
st.title('SVM Prediction App')

# Get the input features from the user
sepal_length = st.number_input('Sepal length')
sepal_width = st.number_input('Sepal width')
petal_length = st.number_input('Petal length')
petal_width = st.number_input('Petal width')

# Create a NumPy array from the input features
X_new = np.array([sepal_length, sepal_width, petal_length, petal_width])

# Make a prediction
prediction = predict(X_new)

# Show the prediction to the user
st.write('Predicted class:', prediction)


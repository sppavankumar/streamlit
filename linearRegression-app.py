import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import streamlit as st

# Title and Description
st.title("Salary Prediction Based on Years of Experience")
st.write("This app uses a linear regression model to predict salary based on years of experience.")

# Loading the dataset
df = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv')
st.write("### Dataset Preview")
st.write(df.head())

# Display scatter plot for data visualization
st.write("### Scatter Plot of Experience vs Salary")
fig, ax = plt.subplots()
ax.scatter(df['Experience Years'], df['Salary'], color='blue')
ax.set_xlabel('Experience Years')
ax.set_ylabel('Salary')
st.pyplot(fig)

# Preparing data for training
x = df[['Experience Years']]
y = df['Salary']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

# Training the Linear Regression model
model = linear_model.LinearRegression()
model.fit(x_train, y_train)

# Making predictions on the test data
y_pred = model.predict(x_test)

# Display actual vs predicted salaries for test set
results_df = pd.DataFrame({"Actual Salary": y_test, "Predicted Salary": y_pred})
st.write("### Actual vs Predicted Salaries on Test Data")
st.write(results_df)

# Add user input for predictions
st.write("### Predict Salary")
experience = st.slider("Select Years of Experience", min_value=0.0, max_value=20.0, step=0.1)
predicted_salary = model.predict([[experience]])[0]
st.write(f"Predicted Salary for {experience} years of experience: ${predicted_salary:.2f}")
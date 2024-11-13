import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Function to load and preprocess data
def load_data():
    # Load data from GitHub
    url = 'https://raw.githubusercontent.com/codebasics/py/801ee0ee4d342fd22b89915dc0c4864250a0ec10/ML/2_linear_reg_multivariate/homeprices.csv'
    df = pd.read_csv(url)
    
    # Fill missing values in 'bedrooms' with the mean
    df['bedrooms'].fillna(df['bedrooms'].mean(), inplace=True)
    return df

# Function to train model
def train_model(df):
    X = df[['area', 'bedrooms', 'age']]
    y = df['price']
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    # Make predictions and calculate metrics
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    return model, x_test, y_test, y_pred, r2, mse

# Streamlit app
def main():
    st.title("Multivariable Linear Regression App")
    st.write("This app demonstrates multivariable linear regression using a housing dataset.")

    # Load and display data
    df = load_data()
    st.write("### Data Preview")
    st.write(df.head())

    # Train model and display results
    if st.button("Train Model"):
        model, x_test, y_test, y_pred, r2, mse = train_model(df)

        # Display model parameters
        st.write("### Model Parameters")
        st.write(f"Intercept: {model.intercept_}")
        st.write(f"Coefficients: {model.coef_}")

        # Display evaluation metrics
        st.write("### Model Evaluation")
        st.write(f"R-squared: {r2}")
        st.write(f"Mean Squared Error: {mse}")

        # Display actual vs. predicted values
        results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.write("### Actual vs. Predicted Values")
        st.write(results)

        # Plot actual vs. predicted values
        st.write("### Prediction Plot")
        st.line_chart(results)

# Run the app
if __name__ == "__main__":
    main()
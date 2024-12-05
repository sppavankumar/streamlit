import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/codebasics/py/refs/heads/master/ML/5_one_hot_encoding/Exercise/carprices.csv"
    return pd.read_csv(url)

# Preprocess data
def preprocess_data(df):
    dummies = pd.get_dummies(df['Car Model'], dtype=int)
    df = pd.concat([df, dummies], axis=1)
    df.drop(columns=['Car Model', 'Mercedez Benz C class'], inplace=True, axis=1)
    return df

# Train model
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Streamlit app
st.title("Car Price Prediction using Linear Regression")
st.markdown("An interactive application to predict car prices using dummy encoding.")

# Load and display data
st.header("Dataset")
df = load_data()
st.dataframe(df)

# Preprocess and display data
st.header("Preprocessed Data")
processed_df = preprocess_data(df)
st.dataframe(processed_df)

# Select features and target
X = processed_df[['Mileage', 'Age(yrs)', 'Audi A5', 'BMW X5']]
y = processed_df['Sell Price($)']

# Train the model
model = train_model(X, y)

# Sidebar inputs for prediction
st.sidebar.header("Input Parameters")
mileage = st.sidebar.number_input("Mileage (in miles):", min_value=0, value=45000)
age = st.sidebar.number_input("Age of Car (in years):", min_value=0, value=4)
is_audi = st.sidebar.selectbox("Is the car Audi A5?", [0, 1])
is_bmw = st.sidebar.selectbox("Is the car BMW X5?", [0, 1])

# Predict and display results
st.header("Prediction")
input_data = [[mileage, age, is_audi, is_bmw]]
prediction = model.predict(input_data)[0]

# Display predicted price
st.markdown(f"""
    <div style="padding: 20px; background-color: #f4f4f4; border-radius: 5px; text-align: center;">
        <h2>Predicted Selling Price:</h2>
        <h1 style="color: #FF6347;"><b>${prediction:,.2f}</b></h1>
    </div>
""", unsafe_allow_html=True)

# Model evaluation (optional for debugging purposes)
mse = mean_squared_error(y, model.predict(X))
r2 = r2_score(y, model.predict(X))
st.sidebar.markdown(f"**Model Metrics:**\n\n- Mean Squared Error: {mse:.2f}\n- R-Squared: {r2:.2f}")

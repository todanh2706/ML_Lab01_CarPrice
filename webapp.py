# app.py (Multi-Model Version)
import streamlit as st
import pandas as pd
import joblib

# --- 1. Load Data ---
@st.cache_data
def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError:
        st.error(f"Data file not found at {filepath}")
        return None

data = load_data('./archive/CarPrice_Assignment.csv')

# --- 2. Model Loading Function ---
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}")
        return None

# --- 3. Sidebar for Inputs ---
st.sidebar.header("Car Details & Model Selection")

# --- Model Selection (Your New Requirement) ---
st.sidebar.subheader("1. Select Model")
model_name = st.sidebar.selectbox(
    "Choose a prediction model:",
    ("Linear Regression", "Random Forest", "Gradient Boosting", "Your Fourth Model")
)

if model_name == "Linear Regression":
    model = load_model('linear_model.joblib')
    st.sidebar.info("Using Linear Regression. Good for baseline.")
elif model_name == "Random Forest":
    model = load_model('random_forest.joblib')
    st.sidebar.info("Using Random Forest. Good for accuracy.")
elif model_name == "Gradient Boosting":
    model = load_model('gradient_boosting.joblib')
    st.sidebar.info("Using Gradient Boosting. Often the most accurate.")
elif model_name == "Your Fourth Model":
    model = load_model('fourth_model.joblib')
    st.sidebar.info("Using Your Fourth Model.")
else:
    st.sidebar.error("Please select a model.")
    st.stop()
    
# --- Feature Inputs ---
st.sidebar.subheader("2. Adjust Features")

horsepower = st.sidebar.slider("Engine Horsepower", 50, 300, 120)
car_width = st.sidebar.slider("Car Width (inches)", 60.0, 80.0, 65.5)
fuel_type = st.sidebar.selectbox("Fuel Type", ("gas", "diesel"))

st.title(f"Car Price Prediction Dashboard")
st.subheader(f"Using Model: {model_name}")

# Create input DataFrame (must match your model's training)
input_data = {
    'horsepower': [horsepower],
    'carwidth': [car_width],
    'fueltype': [fuel_type]
    # ... add all other features
}
input_df = pd.DataFrame(input_data)

st.write("---")

# Create two columns
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Prediction")
    if st.sidebar.button("Predict Price"):
        if model:
            prediction = model.predict(input_df)
            predicted_price = prediction[0]
            
            st.metric(
                label="Predicted Car Price",
                value=f"${predicted_price:,.2f}"
            )
        else:
            st.error("Model is not loaded. Cannot predict.")
            
    # Model Performance (Requirement 4)
    # You would get these values from your notebook
    with st.expander("Show Model Performance"):
        if model_name == "Linear Regression":
            st.write("Model $R^2$: 0.85")
            st.write("Model RMSE: $3,500.00")
        elif model_name == "Random Forest":
            st.write("Model $R^2$: 0.92") 
            st.write("Model RMSE: $2,100.00") 
        # ... etc. for other models

with col2:
    st.header("Data Insights")
    if data is not None:
        st.subheader("Horsepower vs. Price")
        import altair as alt
        chart = alt.Chart(data).mark_circle().encode(
            x='horsepower',
            y='price',
            tooltip=['car_ID', 'CarName', 'horsepower', 'price']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Cannot display chart. Data not loaded.")
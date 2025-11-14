# app.py (Multi-Model Version)
import streamlit as st
import pandas as pd
import joblib
from main1 import TabularPreprocessor, BICForwardSelector
import altair as alt

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
    ("Linear Regression", "Ridge", "Lasso", "Elastic Net")
)

if model_name == "Linear Regression":
    model = load_model('models/linear_bic_pipeline.pkl')
    st.sidebar.info("Using Linear Regression")
elif model_name == "Ridge":
    model = load_model('models/ridge_pipeline.pkl')
    st.sidebar.info("Using Ridge")
elif model_name == "Lasso":
    model = load_model('models/lasso_pipeline.pkl')
    st.sidebar.info("Using Lasso")
elif model_name == "Elastic Net":
    model = load_model('models/elasticnet_pipeline.pkl')
    st.sidebar.info("Using Elastic Net")
else:
    st.sidebar.error("Please select a model.")
    st.stop()
    
st.sidebar.subheader("2. Adjust Features")
    
col1, col2 = st.sidebar.columns(2)

with col1:
    st.write("Categorical Inputs")
    
    symboling = st.selectbox(
        "Symboling", 
        (3, 1, 2, 0, -1, -2)
    )
    
    fueltype = st.selectbox(
        "Fuel Type", 
        ('gas', 'diesel')
    )
    
    # From aspiration (Total Unique: 2)
    aspiration = st.selectbox(
        "Aspiration", 
        ('std', 'turbo')
    )
    
    # From doornumber (Total Unique: 2)
    doornumber = st.selectbox(
        "Door Number", 
        ('two', 'four')
    )
    
    # From carbody (Total Unique: 5)
    carbody = st.selectbox(
        "Car Body", 
        ('convertible', 'hatchback', 'sedan', 'wagon', 'hardtop')
    )
    
    # From drivewheel (Total Unique: 3)
    drivewheel = st.selectbox(
        "Drive Wheel", 
        ('rwd', 'fwd', '4wd')
    )
    
    # From enginelocation (Total Unique: 2)
    enginelocation = st.selectbox(
        "Engine Location", 
        ('front', 'rear')
    )
    
    # From enginetype (Total Unique: 7)
    enginetype = st.selectbox(
        "Engine Type", 
        ('dohc', 'ohcv', 'ohc', 'l', 'rotor', 'ohcf', 'dohcv')
    )
    
    # From cylindernumber (Total Unique: 7)
    cylindernumber = st.selectbox(
        "Cylinder Number", 
        ('four', 'six', 'five', 'three', 'twelve', 'two', 'eight')
    )
    
    # From fuelsystem (Total Unique: 8)
    fuelsystem = st.selectbox(
        "Fuel System", 
        ('mpfi', '2bbl', 'mfi', '1bbl', 'spfi', '4bbl', 'idi', 'spdi')
    )

    # For CarName -> CompanyName (Engineered Feature)
    CompanyName = st.text_input("Company Name", "audi")
    st.caption("Replace this with an `st.selectbox` using your unique company list for better results.")


# === COLUMN 2: NUMERIC FEATURES ===
with col2:
    st.write("Numeric Inputs")
    
    wheelbase = st.slider("Wheelbase (in)", 85.0, 125.0, 99.8)
    
    carlength = st.slider("Car Length (in)", 140.0, 210.0, 176.6)
    
    carwidth = st.slider("Car Width (in)", 60.0, 75.0, 66.2)
    
    carheight = st.slider("Car Height (in)", 47.0, 60.0, 54.3)
    
    curbweight = st.slider("Curb Weight (lbs)", 1400, 4100, 2500)
    
    enginesize = st.slider("Engine Size (cu in)", 60, 330, 130)
    
    boreratio = st.slider("Bore Ratio", 2.50, 4.00, 3.19)
    
    stroke = st.slider("Stroke (in)", 2.00, 4.20, 3.40)
    
    compressionratio = st.slider("Compression Ratio", 7.0, 23.0, 9.0)
    
    horsepower = st.slider("Horsepower", 40, 290, 110)
    
    peakrpm = st.slider("Peak RPM", 4000, 7000, 5500)
    
    citympg = st.slider("City MPG", 13, 50, 24)
    
    highwaympg = st.slider("Highway MPG", 16, 55, 30)

st.title(f"Car Price Prediction Dashboard")
st.subheader(f"Using Model: {model_name}")

input_data = {
    # Numerical
    'wheelbase': wheelbase,
    'carlength': carlength,
    'carwidth': carwidth,
    'carheight': carheight,
    'curbweight': curbweight,
    'enginesize': enginesize,
    'boreratio': boreratio,
    'stroke': stroke,
    'compressionratio': compressionratio,
    'horsepower': horsepower,
    'peakrpm': peakrpm,
    'citympg': citympg,
    'highwaympg': highwaympg,
    
    # Categorical
    'fueltype': fueltype,
    'aspiration': aspiration,
    'doornumber': doornumber,
    'carbody': carbody,
    'drivewheel': drivewheel,
    'enginelocation': enginelocation,
    
    # --- Add the rest of your features ---
    'enginetype': enginetype,
    'cylindernumber': cylindernumber,
    'fuelsystem': fuelsystem,
    'symboling': symboling,
    'CompanyName': CompanyName
}
input_df = pd.DataFrame([input_data])

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
            

with col2:
    st.header("Data Insights")
    st.subheader(f"Feature Influence for {model_name}")
    
    if model:
        try:
            # Extract the coef
            coefficients = model.named_steps['linear_model'].coef_
            
            # Extract the feature names
            if model_name == "Linear Regression":
                feature_names = model.named_steps['bic_selector'].get_feature_names_out()
            else:
                feature_names = model.named_steps['preprocess'].feature_columns_

            # Ensure the equality between the number of coefs. and the number of feature names
            if len(coefficients) == len(feature_names):
                # Create df
                coef_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Influence': coefficients
                })
                
                # Coloring
                coef_df['Positive'] = coef_df['Influence'] > 0
                
                # Sort by absoluted value to find the most influencer
                coef_df['Absolute_Influence'] = coef_df['Influence'].abs()
                coef_df = coef_df.sort_values(by='Absolute_Influence', ascending=False)
                
                # Top 20
                top_n = 20
                top_n_actual = min(top_n, len(coef_df)) 
                
                coef_df_top = coef_df.head(top_n_actual).sort_values(by='Influence', ascending=False)

                # Draw usin Altair
                chart = alt.Chart(coef_df_top).mark_bar().encode(
                    # Sort Y-axis
                    x=alt.X('Influence:Q'),
                    y=alt.Y('Feature:N', sort=alt.SortField(field="Influence", order='descending')),
                    
                    # Coloring
                    color=alt.Color('Positive:N', legend=None, scale={'range': ['#FF6B6B', '#6BFFB8']}),
                    tooltip=['Feature', 'Influence']
                ).properties(
                    title=f"Top {top_n_actual} Feature Influences ({model_name})"
                ).interactive()
                
                st.altair_chart(chart, use_container_width=True)
                
                # Add expander for more detail viewing
                with st.expander("See all feature coefficients"):
                    st.dataframe(coef_df.sort_values(by='Influence', ascending=False))
            
            else:
                st.error(f"Error: The number of feature ({len(feature_names)}) and the coefs ({len(coefficients)}) do not match.")
                
        except KeyError as e:
            st.error(f"Error: Cannot find the pipeline step: {e}.")
        except Exception as e:
            st.error(f"An error occures when getting coef: {e}")
    else:
        st.info("Model is not loaded.")

    st.write("---")

    if data is not None:
        st.subheader("Horsepower vs. Price (Raw Data)")
        chart = alt.Chart(data).mark_circle().encode(
            x='horsepower',
            y='price',
            tooltip=['car_ID', 'CarName', 'horsepower', 'price']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Cannot display chart. Data not loaded.")
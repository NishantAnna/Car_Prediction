import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# --- Load Model Artifacts ---
@st.cache_resource
def load_models_and_data():
    """Loads all necessary model artifacts and data."""
    try:
        pipeline = joblib.load("car_price_model.pkl")
        kmeans = joblib.load("kmeans_region.pkl")
        raw_training_columns = joblib.load("raw_training_columns.pkl")
        df_full = pd.read_csv("Final_Used_Cars.csv")
        return pipeline, kmeans, raw_training_columns, df_full
    except FileNotFoundError as e:
        st.error(f"Error: Required file not found. Please ensure {e.filename} is in the same directory.")
        return None, None, None, None

pipeline, kmeans, raw_training_columns, df_full = load_models_and_data()

if pipeline is None:
    st.stop()

# --- Create Mappings for User-Friendly Input and Output ---
# NOTE: These mappings are hardcoded for simplicity. In a real-world app,
# you would save these mappings from your training script.
make_map = {3: "Ford", 6: "Chevrolet", 7: "Honda", 8: "Nissan", 9: "Toyota"}
model_map = {71: "F-150", 17: "Camaro", 51: "Civic"}
trim_map = {11: "XLT", 45: "LT", 29: "Touring"}
transmission_map = {0: "Automatic", 1: "Manual"}
drivetrain_map = {0: "4WD", 1: "RWD", 2: "FWD"}

# Get unique integer values for dropdowns from the loaded data
make_options = sorted(make_map.keys())
model_options = sorted(model_map.keys())
trim_options = sorted(trim_map.keys())
transmission_options = sorted(transmission_map.keys())
drivetrain_options = sorted(drivetrain_map.keys())

# --- App Title and Description ---
st.title("🚗 Used Car Price Predictor")
st.markdown("Enter the details of a used car to get a predicted price.")
st.markdown("---")

# --- User Input Widgets ---
st.header("Car Details")

# Use a two-column layout for better aesthetics
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", min_value=1980, max_value=datetime.now().year, value=2018)
    # Use user-friendly names for selection, and get the numeric label
    make_name = st.selectbox("Make", list(make_map.values()), index=list(make_map.values()).index("Ford"))
    model_name = st.selectbox("Model", list(model_map.values()), index=list(model_map.values()).index("F-150"))
    trim_name = st.selectbox("Trim", list(trim_map.values()), index=list(trim_map.values()).index("XLT"))
    miles = st.number_input("Miles", min_value=0, value=50000)
    transmission_name = st.selectbox("Transmission", list(transmission_map.values()))
    drivetrain_name = st.selectbox("Drivetrain", list(drivetrain_map.values()))

with col2:
    highway_mpg = st.number_input("Highway MPG", min_value=0, value=25)
    city_mpg = st.number_input("City MPG", min_value=0, value=20)
    doors = st.number_input("Doors", min_value=2, max_value=5, value=4)
    latitude = st.number_input("Latitude", value=44.71164)
    longitude = st.number_input("Longitude", value=-92.851607)

# --- Prediction Button ---
if st.button("Predict Price"):
    # Convert user-friendly names back to numerical labels for the model
    make_val = [k for k, v in make_map.items() if v == make_name][0]
    model_val = [k for k, v in model_map.items() if v == model_name][0]
    trim_val = [k for k, v in trim_map.items() if v == trim_name][0]
    transmission_val = [k for k, v in transmission_map.items() if v == transmission_name][0]
    drivetrain_val = [k for k, v in drivetrain_map.items() if v == drivetrain_name][0]

    # Create a DataFrame from user input
    input_data = {
        "year": year, "make": make_val, "model": model_val, "trim": trim_val,
        "miles": miles, "transmission": transmission_val, "drivetrain": drivetrain_val,
        "highway_mpg": highway_mpg, "city_mpg": city_mpg, "doors": doors,
        "latitude": latitude, "longitude": longitude
    }
    input_df = pd.DataFrame([input_data])

    # Replicate the Feature Engineering steps from the training script
    current_year = datetime.now().year
    input_df["car_age"] = current_year - input_df["year"]
    input_df["car_age"] = input_df["car_age"].clip(lower=0)
    input_df["age_miles_ratio"] = input_df["miles"] / (input_df["car_age"] + 1)
    input_df["age_miles_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)
    input_df["age_miles_ratio"].fillna(0, inplace=True)
    input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    input_df.fillna(0, inplace=True)
    input_df["mileage_bucket"] = pd.cut(
        input_df["miles"],
        bins=[-1, 20000, 60000, 100000, np.inf],
        labels=["Low", "Medium", "High", "Very_High"]
    )
    coords = input_df[["latitude", "longitude"]].fillna(0).to_numpy()
    input_df["region_cluster"] = kmeans.predict(coords).astype(str)

    # Ensure the columns are in the correct order for the pipeline
    for col in raw_training_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[raw_training_columns]

    # Make a prediction
    predicted_price = pipeline.predict(input_df)[0]
    
    st.markdown("---")
    st.subheader("Predicted Price:")
    st.metric(label="Estimated Car Price", value=f"${predicted_price:,.2f}")
    
    # Display the full details of the car for the user
    st.markdown("### Car Details Entered")
    st.write(f"**Year:** {year}")
    st.write(f"**Make:** {make_name}")
    st.write(f"**Model:** {model_name}")
    st.write(f"**Trim:** {trim_name}")
    st.write(f"**Miles:** {miles:,.0f}")
    st.write(f"**Transmission:** {transmission_name}")
    st.write(f"**Drivetrain:** {drivetrain_name}")
    st.write(f"**Highway MPG:** {highway_mpg}")
    st.write(f"**City MPG:** {city_mpg}")
    st.write(f"**Doors:** {doors}")
    st.write(f"**Latitude:** {latitude}")
    st.write(f"**Longitude:** {longitude}")
    
    st.balloons()

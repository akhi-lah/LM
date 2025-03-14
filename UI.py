import streamlit as st
import pandas as pd
from model import load_data, preprocess_data, load_model, assign_workers_to_regions
import time

# Set up Streamlit UI
st.set_page_config(page_title="Warehouse Worker Assignment Tool", layout="wide")
st.title("Warehouse Worker Assignment Optimizer")

# Cache the model so it's loaded only once per session - improved with TTL
@st.cache_resource(ttl=3600)
def get_model():
    return load_model()

# Cache the data to avoid reloading it unnecessarily - improved with TTL
@st.cache_data(ttl=3600)
def get_data():
    start = time.time()
    df = load_data()
    df, features, targets = preprocess_data(df)
    print(f"Data loading and preprocessing took {time.time() - start:.2f} seconds")
    return df

# Load model and data
model = get_model()
df = get_data()

# Temperature and Humidity Inputs
cols = st.columns(2)
with cols[0]:
    temp = st.slider("Temperature (°C)", min_value=-50.0, max_value=45.0, value=20.0, step=0.5)
with cols[1]:
    humidity = st.slider("Humidity (%)", min_value=20.0, max_value=90.0, value=50.0, step=1.0)

# Warn if temperature is outside training range
if temp < df["Temperature"].min() or temp > df["Temperature"].max():
    st.warning("⚠️ Temperature is outside the training range. Predictions may be inaccurate.")

# Zone Filter Section
st.subheader("Zone Filter")
num_zones = st.number_input("Number of Zones", min_value=1, max_value=10, value=3, step=1)
zone_data = []
zone_cols = st.columns(min(num_zones, 4))
for i in range(num_zones):
    zone = i + 1
    with zone_cols[i % 4]:
        quantity = st.number_input(f"Zone {zone} Quantity", min_value=1, max_value=1000, value=100, step=10, key=f"zone_{zone}")
        zone_data.append({"zone": zone, "quantity": quantity})
zone_df = pd.DataFrame(zone_data)

# Cache the optimization function to avoid rerunning with the same inputs
@st.cache_data(ttl=600)
def cached_optimization(zone_df_json, temp_val, humidity_val):
    zone_df_local = pd.read_json(zone_df_json)
    return assign_workers_to_regions(zone_df_local, df, temp_val, humidity_val)

# Run Optimization
if st.button("Run Optimization", type="primary"):
    # Create a placeholder for progress messages
    status = st.empty()
    status.text("Processing... This might take a few moments.")
    
    # Convert DataFrame to JSON for caching purpose
    zone_df_json = zone_df.to_json()
    
    try:
        # Use the cached optimization function
        with st.spinner("Running optimization..."):
            start_time = time.time()
            result_df = cached_optimization(zone_df_json, temp, humidity)
            opt_time = time.time() - start_time
            
        # Show success message with timing
        status.success(f"Optimization completed in {opt_time:.2f} seconds")
        
        # Process the result dataframe (same as original)
        formatted_results = [
            {
                "Zone": f"Zone {row['Zone']}",
                "Proposed Team": detail["worker_id"],
                "Processed Quantity": row['Processed_Quantity'],
                "Team Size": row['Team Size'],
                "Individual ETC": detail["Individual ETC"],
                "Individual Productivity": round(float(str(detail["Individual Productivity"]).replace(" items/hr", ""))),
                "Team ETC": row['EstimatedTimeToPickTheQuantity'],
                "Team Productivity": round(float(str(row['Team Productivity']).replace(" items/hr", "")))
            }
            for _, row in result_df.iterrows()
            for detail in row['WorkerDetails']
        ]
        formatted_df = pd.DataFrame(formatted_results)
        
        # Display the results as a formatted table
        st.subheader("Worker Assignments")
        st.dataframe(formatted_df)
        
    except Exception as e:
        status.error(f"Error during optimization: {str(e)}")

# Style Enhancements
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4e8cff;
        color: white;
        font-weight: bold;
    }
    .stDataFrame {
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

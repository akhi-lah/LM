import streamlit as st
import pandas as pd
from model import load_data, preprocess_data, load_model, assign_workers_to_regions
import time

# Set up Streamlit UI
st.set_page_config(page_title="Warehouse Worker Assignment Tool", layout="wide")
st.title("Warehouse Worker Assignment Optimizer")

# Performance optimizations
# 1. Improve caching strategy
@st.cache_resource(ttl=3600)
def get_model():
    return load_model()

@st.cache_data(ttl=3600)
def get_data():
    df = load_data()
    df, *_ = preprocess_data(df)
    return df

# 2. Add caching for optimization results
@st.cache_data(ttl=600, show_spinner=False)
def run_optimization(zone_df, base_df, temperature, humidity):
    return assign_workers_to_regions(zone_df, base_df, temperature, humidity)

# Load model and data with progress indicator
with st.spinner("Loading model and data..."):
    start_time = time.time()
    model = get_model()
    df = get_data()
    load_time = time.time() - start_time
    st.success(f"Model and data loaded in {load_time:.2f} seconds")

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

# 3. Use session state to persist zone data between reruns
if 'zone_data' not in st.session_state:
    st.session_state.zone_data = [{"zone": 1, "quantity": 100}, {"zone": 2, "quantity": 100}, {"zone": 3, "quantity": 100}]

num_zones = st.number_input("Number of Zones", min_value=1, max_value=10, value=3, step=1)

# Adjust session state if number of zones changes
if len(st.session_state.zone_data) != num_zones:
    if len(st.session_state.zone_data) < num_zones:
        # Add new zones
        for i in range(len(st.session_state.zone_data), num_zones):
            st.session_state.zone_data.append({"zone": i + 1, "quantity": 100})
    else:
        # Remove excess zones
        st.session_state.zone_data = st.session_state.zone_data[:num_zones]

# Display zone inputs in a grid
zone_cols = st.columns(min(num_zones, 4))
for i in range(num_zones):
    with zone_cols[i % 4]:
        st.session_state.zone_data[i]["quantity"] = st.number_input(
            f"Zone {i+1} Quantity", 
            min_value=1, 
            max_value=1000, 
            value=st.session_state.zone_data[i]["quantity"], 
            step=10, 
            key=f"zone_{i+1}"
        )

# Create dataframe from session state
zone_df = pd.DataFrame(st.session_state.zone_data)

# 4. Add a form to prevent rerunning on every change
with st.form("optimization_form"):
    submit_button = st.form_submit_button("Run Optimization", type="primary")

# Process when the form is submitted
if submit_button:
    # Display a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run optimization with caching
    status_text.text("Starting optimization...")
    progress_bar.progress(10)
    
    start_time = time.time()
    
    try:
        # Break down the process into phases for visual feedback
        status_text.text("Processing data...")
        progress_bar.progress(30)
        
        # Run the optimization with a timeout
        result_df = run_optimization(zone_df, df, temp, humidity)
        
        status_text.text("Formatting results...")
        progress_bar.progress(70)
        
        # Process the result dataframe
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
        
        progress_bar.progress(90)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Display the results
        formatted_df = pd.DataFrame(formatted_results)
        status_text.text(f"Optimization completed in {process_time:.2f} seconds")
        progress_bar.progress(100)
        
        # Display the results as a formatted table
        st.subheader("Worker Assignments")
        st.dataframe(formatted_df)
        
    except Exception as e:
        st.error(f"An error occurred during optimization: {str(e)}")
        progress_bar.empty()
        status_text.empty()

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

# Add additional performance info
with st.expander("Performance Information"):
    st.info("""
    This application uses advanced caching to improve performance:
    - Model and data are cached for 1 hour
    - Optimization results are cached for 10 minutes
    - Form submission prevents unnecessary recalculations
    """)

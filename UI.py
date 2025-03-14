import streamlit as st
import pandas as pd
from model import load_data, preprocess_data, load_model, assign_workers_to_regions
import time

# Set up Streamlit UI
st.set_page_config(page_title="Warehouse Worker Assignment Tool", layout="wide")
st.title("Warehouse Worker Assignment Optimizer")

# Performance tracking
start_time = time.time()

# Cache the model and data with proper TTL and hash_funcs for stability
@st.cache_resource(ttl=3600)
def get_model():
    return load_model()

@st.cache_data(ttl=3600)
def get_data():
    df = load_data()
    processed_df, features, targets = preprocess_data(df)
    return processed_df, features, targets

# Load model and data with progress indicator
with st.spinner("Loading model and data..."):
    model = get_model()
    df, features, targets = get_data()
    load_time = time.time() - start_time
    st.success(f"Data and model loaded in {load_time:.2f} seconds")

# Create sidebar for inputs to declutter main area
with st.sidebar:
    st.header("Environmental Parameters")
    temp = st.slider("Temperature (°C)", min_value=-50.0, max_value=45.0, value=20.0, step=0.5)
    humidity = st.slider("Humidity (%)", min_value=20.0, max_value=90.0, value=50.0, step=1.0)
    
    # Warn if temperature is outside training range
    if temp < df["Temperature"].min() or temp > df["Temperature"].max():
        st.warning("⚠️ Temperature is outside the training range. Predictions may be inaccurate.")

# Zone configuration in the main area with a more compact layout
st.subheader("Zone Configuration")

# Use columns for a more compact zone input layout
col1, col2 = st.columns([1, 3])
with col1:
    num_zones = st.number_input("Number of Zones", min_value=1, max_value=10, value=3, step=1)

# Create zone inputs in a grid layout
zone_data = []
rows = (num_zones + 3) // 4  # Calculate number of rows needed (4 zones per row)
for row in range(rows):
    cols = st.columns(4)
    for i in range(4):
        idx = row * 4 + i
        if idx < num_zones:
            zone = idx + 1
            with cols[i]:
                quantity = st.number_input(
                    f"Zone {zone}", 
                    min_value=1, 
                    max_value=1000, 
                    value=100, 
                    step=10, 
                    key=f"zone_{zone}"
                )
                zone_data.append({"zone": zone, "quantity": quantity})

zone_df = pd.DataFrame(zone_data)

# Run Optimization with progress tracking
if st.button("Run Optimization", type="primary"):
    opt_start_time = time.time()
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Split the optimization into steps for better user experience
    status_text.text("Preparing optimization...")
    progress_bar.progress(10)
    
    # Use st.cache_data for the assign_workers function to avoid redundant calculations
    @st.cache_data(ttl=600, show_spinner=False)
    def cached_assign_workers(zone_df_json, temp, humidity):
        # Convert the JSON string back to DataFrame
        zone_df_loaded = pd.read_json(zone_df_json)
        return assign_workers_to_regions(zone_df_loaded, df, temp, humidity)
    
    # Update progress
    status_text.text("Running optimization algorithm...")
    progress_bar.progress(30)
    
    # Convert DataFrame to JSON string for caching
    zone_df_json = zone_df.to_json()
    
    try:
        # Run the optimization with caching
        result_df = cached_assign_workers(zone_df_json, temp, humidity)
        
        # Update progress
        status_text.text("Processing results...")
        progress_bar.progress(70)
        
        # Process results more efficiently
        formatted_results = []
        for _, row in result_df.iterrows():
            zone = row['Zone']
            team_size = row['Team Size']
            processed_qty = row['Processed_Quantity']
            team_etc = row['EstimatedTimeToPickTheQuantity']
            team_prod = row['Team Productivity']
            
            # Extract the numeric value from team productivity string
            if isinstance(team_prod, str) and "items/hr" in team_prod:
                team_prod = float(team_prod.replace(" items/hr", ""))
            
            for detail in row['WorkerDetails']:
                worker_id = detail["worker_id"]
                individual_etc = detail["Individual ETC"]
                individual_prod = detail["Individual Productivity"]
                
                # Extract the numeric value from individual productivity string
                if isinstance(individual_prod, str) and "items/hr" in individual_prod:
                    individual_prod = float(individual_prod.replace(" items/hr", ""))
                
                formatted_results.append({
                    "Zone": f"Zone {zone}",
                    "Proposed Team": worker_id,
                    "Processed Quantity": processed_qty,
                    "Team Size": team_size,
                    "Individual ETC": individual_etc,
                    "Individual Productivity": round(float(individual_prod)),
                    "Team ETC": team_etc,
                    "Team Productivity": round(float(team_prod))
                })
        
        formatted_df = pd.DataFrame(formatted_results)
        
        # Update progress
        progress_bar.progress(100)
        opt_time = time.time() - opt_start_time
        status_text.success(f"Optimization completed in {opt_time:.2f} seconds")
        
        # Display the results as a formatted table with sorting enabled
        st.subheader("Worker Assignments")
        st.dataframe(
            formatted_df,
            use_container_width=True,
            column_config={
                "Individual Productivity": st.column_config.NumberColumn(
                    "Individual Productivity",
                    format="%d items/hr"
                ),
                "Team Productivity": st.column_config.NumberColumn(
                    "Team Productivity",
                    format="%d items/hr"
                )
            }
        )
        
        # Add download button for results
        csv = formatted_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="worker_assignments.csv",
            mime="text/csv",
        )
        
    except Exception as e:
        st.error(f"An error occurred during optimization: {str(e)}")
        st.exception(e)

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
    .sidebar .sidebar-content {
        background-color: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Add a footer with execution information
total_exec_time = time.time() - start_time
st.caption(f"Total app execution time: {total_exec_time:.2f} seconds")

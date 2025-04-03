import streamlit as st
import pandas as pd
import os
import ast  # for safe evaluation of string representations
from model import load_model, assign_workers_to_zones, preprocess_data, load_data, categorize_temperature

# Set page configuration
st.set_page_config(page_title="Warehouse Worker Assignment", layout="wide")

# Ensure models exist
ambient_model_path = "ambient_model.pkl"
cold_cooler_model_path = "cold_cooler_model.pkl"

if not os.path.exists(ambient_model_path) or not os.path.exists(cold_cooler_model_path):
    st.error("One or both model files are missing. Run model.py to train the models.")
    st.stop()

# Load trained models into session state
if 'ambient_model' not in st.session_state:
    st.session_state.ambient_model = load_model(ambient_model_path)
if 'cold_cooler_model' not in st.session_state:
    st.session_state.cold_cooler_model = load_model(cold_cooler_model_path)

st.sidebar.title("Warehouse Worker Assignment")

# Load dataset for worker data
df = load_data()
df, le = preprocess_data(df)

# Show temperature distribution if Room_Temp column exists
if 'Room_Temp' in df.columns:
    # Calculate temperature ranges for display
    temp_ranges = {
        'Cold': df[df['Room_Temp'] < 0]['Room_Temp'].describe(),
        'Cool': df[(df['Room_Temp'] >= 0) & (df['Room_Temp'] <= 10)]['Room_Temp'].describe(),
        'Ambient': df[df['Room_Temp'] > 10]['Room_Temp'].describe()
    }

# Input fields for user parameters
st.sidebar.subheader("Input Warehouse Parameters")
ambient_temp = st.sidebar.number_input("Ambient Temperature (°C)", min_value=-10, max_value=50, value=15)
st.sidebar.write(f"Temperature Category: {categorize_temperature(ambient_temp)}")

ambient_humidity = st.sidebar.number_input("Ambient Humidity (%)", min_value=0, max_value=100, value=60)
ambient_qty = st.sidebar.number_input("Quantity to Pick (Ambient)", min_value=1, value=500)
cold_qty = st.sidebar.number_input("Quantity to Pick (Cold)", min_value=1, value=800)
cooler_qty = st.sidebar.number_input("Quantity to Pick (Cooler)", min_value=1, value=1000)

# Prepare user inputs
user_inputs = {
    "Ambient_temp": ambient_temp,
    "Ambient_humidity": ambient_humidity,
    "Ambient_qty": ambient_qty,
    "Cold_qty": cold_qty,
    "Cooler_qty": cooler_qty,
}

# Create a DataFrame for zone input
zone_input = pd.DataFrame({"zone": ["Ambient", "Cold", "Cooler"]})

# Run assignment only when the button is clicked; store results in session state.
if st.sidebar.button("Assign Workers"):
    with st.spinner("Processing..."):
        result_df = assign_workers_to_zones(zone_input, df, le, user_inputs)
        st.session_state["result_df"] = result_df  # store result for later use in simulation
    st.success("Worker assignment completed successfully!")

# Display zone temperature definitions
st.subheader("Temperature Zone Definitions")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='background-color: #d9edf7; border: 2px solid #31708f; border-radius: 10px; padding: 10px;'>
        <h3 style='text-align: center; color: #31708f;'>Cold Zone</h3>
        <p>Temperature: < 0°C</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background-color: #dff0d8; border: 2px solid #3c763d; border-radius: 10px; padding: 10px;'>
        <h3 style='text-align: center; color: #3c763d;'>Cool Zone</h3>
        <p>Temperature: 0°C to 10°C</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='background-color: #fcf8e3; border: 2px solid #8a6d3b; border-radius: 10px; padding: 10px;'>
        <h3 style='text-align: center; color: #8a6d3b;'>Ambient Zone</h3>
        <p>Temperature: > 10°C</p>
    </div>
    """, unsafe_allow_html=True)

# If Room_Temp exists, display temperature stats
if 'Room_Temp' in df.columns:
    #st.subheader("Temperature Distribution in Dataset")
    temp_stats_cols = st.columns(3)
    
    # with temp_stats_cols[0]:
    #     cold_count = len(df[df['Room_Temp'] < 0])
    #     st.metric("Cold Zone Workers", cold_count)
    #     if 'Cold' in temp_ranges and not temp_ranges['Cold'].empty:
    #         st.write(f"Min: {temp_ranges['Cold']['min']:.1f}°C, Max: {temp_ranges['Cold']['max']:.1f}°C")
    
    # with temp_stats_cols[1]:
    #     cool_count = len(df[(df['Room_Temp'] >= 0) & (df['Room_Temp'] <= 10)])
    #     st.metric("Cool Zone Workers", cool_count)
    #     if 'Cool' in temp_ranges and not temp_ranges['Cool'].empty:
    #         st.write(f"Min: {temp_ranges['Cool']['min']:.1f}°C, Max: {temp_ranges['Cool']['max']:.1f}°C")
    
    # with temp_stats_cols[2]:
    #     ambient_count = len(df[df['Room_Temp'] > 10])
    #     st.metric("Ambient Zone Workers", ambient_count)
    #     if 'Ambient' in temp_ranges and not temp_ranges['Ambient'].empty:
    #         st.write(f"Min: {temp_ranges['Ambient']['min']:.1f}°C, Max: {temp_ranges['Ambient']['max']:.1f}°C")

# If assignment results are available, display the table and simulation section.
if "result_df" in st.session_state:
    result_df = st.session_state.result_df
    
    if "RESOURCE" in df.columns:
        total_workers = df["RESOURCE"].astype(str).str.strip().nunique()
    elif "worker_id" in df.columns:
        total_workers = df["worker_id"].astype(str).str.strip().nunique()
    elif "resource" in df.columns:  # Add this case
        total_workers = df["resource"].astype(str).str.strip().nunique()
    else:
        # If there's some other worker identifier column, use that
        worker_id_col = [col for col in df.columns if 'worker' in col.lower() or 'resource' in col.lower()]
        if worker_id_col:
            total_workers = df[worker_id_col[0]].astype(str).str.strip().nunique()
        else:
            total_workers = 30  # Fallback to your known count
    
    # Calculate assigned workers from the assignment result.
    assigned_workers_set = set()
    for _, row in result_df.iterrows():
        worker_details = row.get("WorkerDetails", [])
        if isinstance(worker_details, str):
            worker_details = ast.literal_eval(worker_details)
        for detail in worker_details:
            if isinstance(detail, dict) and "worker_id" in detail:
                assigned_workers_set.add(detail["worker_id"].strip() if isinstance(detail["worker_id"], str) else detail["worker_id"])
    assigned_workers = len(assigned_workers_set)
    
    # Display heading and two styled number cards for total and assigned workers.
    st.subheader("Worker Assignments")
    col1, col2 = st.columns(2)
    
    card_style_total = """
    <div style='background-color: #f0f8ff; border: 2px solid #007acc; border-radius: 10px; padding: 20px; text-align: center;'>
        <h2 style='font-size: 24px; margin: 0;'>TOTAL WORKERS</h2>
        <p style='font-size: 30px; font-weight: bold; margin: 0;'>{value}</p>
    </div>
    """
    card_style_assigned = """
    <div style='background-color: #f0f8ff; border: 2px solid #007acc; border-radius: 10px; padding: 20px; text-align: center;'>
        <h2 style='font-size: 24px; margin: 0;'>ASSIGNED WORKERS</h2>
        <p style='font-size: 30px; font-weight: bold; margin: 0;'>{value}</p>
    </div>
    """
    with col1:
        st.markdown(card_style_total.format(value=total_workers), unsafe_allow_html=True)
    with col2:
        st.markdown(card_style_assigned.format(value=assigned_workers), unsafe_allow_html=True)
    
    # Display the assignment results in an HTML table.
    formatted_results = []
    for _, row in result_df.iterrows():
        zone = row.get('Zone')
        processed_quantity = row.get('Processed_Quantity')
        team_size = row.get('Team Size')
        team_etc = row.get('EstimatedTimeToPickTheQuantity')
        team_productivity = row.get('Team Productivity')
        # Round team productivity if it is a number
        try:
            team_productivity = f"{round(float(team_productivity))} items/hr"
        except:
            team_productivity = team_productivity

        worker_details = row.get('WorkerDetails', [])
        if isinstance(worker_details, str):
            worker_details = ast.literal_eval(worker_details)
            
        for detail in worker_details:
            if isinstance(detail, dict):
                # Process and round individual productivity
                ind_prod_str = detail.get("Individual Productivity", "0")
                try:
                    ind_prod_value = float(ind_prod_str.split()[0])
                except:
                    ind_prod_value = float(ind_prod_str)
                ind_prod_rounded = f"{round(ind_prod_value)} items/hr"
                
                formatted_results.append({
                    "Zone": f"Zone {zone}",
                    "Proposed Team": detail.get("worker_id", "Unknown"),
                    "Processed Quantity": processed_quantity,
                    "Team Size": team_size,
                    "Individual ETC": detail.get("Individual ETC", "N/A"),
                    "Individual Productivity (items/hr)": ind_prod_rounded,
                    "Team ETC": team_etc,
                    "Team Productivity (items/hr)": team_productivity
                })
    
    if formatted_results:
        formatted_df = pd.DataFrame(formatted_results)
        html_table = """
        <div style="overflow-x: auto;">
        <table style="width:100%; border-collapse: collapse; margin-top: 20px;">
            <thead>
                <tr style="background-color: #007acc; color: white;">
                    <th style="padding: 10px; border: 1px solid #ddd;">Zone</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Proposed Team</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Processed Quantity</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Team Size</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Individual ETC</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Individual Productivity (items/hr)</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Team ETC</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">Team Productivity (items/hr)</th>
                </tr>
            </thead>
            <tbody>
        """
        grouped_df = formatted_df.groupby("Zone")
        for zone, group in grouped_df:
            rows = len(group)
            first_row = True
            for _, row in group.iterrows():
                html_table += "<tr style='background-color: " + ("#f0f8ff" if first_row else "#ffffff") + ";'>"
                if first_row:
                    html_table += f"<td style='padding: 10px; border: 1px solid #ddd;' rowspan='{rows}'>{zone}</td>"
                    html_table += f"<td style='padding: 10px; border: 1px solid #ddd;'>{row['Proposed Team']}</td>"
                    html_table += f"<td style='padding: 10px; border: 1px solid #ddd;' rowspan='{rows}'>{row['Processed Quantity']}</td>"
                    html_table += f"<td style='padding: 10px; border: 1px solid #ddd;' rowspan='{rows}'>{row['Team Size']}</td>"
                    html_table += f"<td style='padding: 10px; border: 1px solid #ddd;'>{row['Individual ETC']}</td>"
                    html_table += f"<td style='padding: 10px; border: 1px solid #ddd;'>{row['Individual Productivity (items/hr)']}</td>"
                    html_table += f"<td style='padding: 10px; border: 1px solid #ddd;' rowspan='{rows}'>{row['Team ETC']}</td>"
                    html_table += f"<td style='padding: 10px; border: 1px solid #ddd;' rowspan='{rows}'>{row['Team Productivity (items/hr)']}</td>"
                else:
                    html_table += f"<td style='padding: 10px; border: 1px solid #ddd;'>{row['Proposed Team']}</td>"
                    html_table += f"<td style='padding: 10px; border: 1px solid #ddd;'>{row['Individual ETC']}</td>"
                    html_table += f"<td style='padding: 10px; border: 1px solid #ddd;'>{row['Individual Productivity (items/hr)']}</td>"
                html_table += "</tr>"
                first_row = False
        html_table += """
            </tbody>
        </table>
        </div>
        """
        st.markdown(html_table, unsafe_allow_html=True)
    
    # Simulation: Let user select a zone and then pick workers (default unselected).
    st.subheader("Zone Worker Combination Simulation")
    # Zone selectbox (outside any form so it updates immediately)
    selected_zone = st.selectbox("Select Zone", options=result_df["Zone"].unique(), key="simulation_zone")
    zone_data = result_df[result_df["Zone"] == selected_zone].iloc[0]
    qty = zone_data["Processed_Quantity"]
    # Process WorkerDetails: if it's a string, convert it.
    worker_details = zone_data["WorkerDetails"]
    if isinstance(worker_details, str):
        worker_details = ast.literal_eval(worker_details)
    worker_options = [wd["worker_id"] for wd in worker_details]
    
    # Use a form so that changes in the multiselect do not trigger an automatic re-run.
    with st.form("simulation_form"):
        selected_workers = st.multiselect("Select Workers for Simulation", options=worker_options, default=[], key="simulation_workers")
        calculate_clicked = st.form_submit_button("Calculate")
        
        if calculate_clicked:
            if selected_workers:
                total_productivity = 0
                for wd in worker_details:
                    if wd["worker_id"] in selected_workers:
                        # Convert string like "596.46 items/hr" to float
                        prod_str = wd["Individual Productivity"]
                        prod_value = float(prod_str.split()[0])
                        total_productivity += prod_value
                if total_productivity > 0:
                    estimated_time_minutes = (qty / total_productivity) * 60
                    hours = int(estimated_time_minutes // 60)
                    minutes = int(estimated_time_minutes % 60)
                    st.write(f"Estimated time for the selected team: {hours} hrs {minutes} mins")
                else:
                    st.write("Selected team has zero productivity.")
            else:
                st.write("No workers selected for simulation.")

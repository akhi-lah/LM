import streamlit as st
import pandas as pd
from model import load_data, preprocess_data, load_model, assign_workers_to_regions

# Use session state to cache data
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.preprocessed_df = None

st.set_page_config(page_title="Warehouse Worker Assignment Tool", layout="wide")
st.title("Warehouse Worker Assignment Optimizer")

# Load data and model only once when the app starts
if not st.session_state.data_loaded:
    with st.spinner("Loading data and model..."):
        # Load the pre-trained model instead of retraining
        load_model()  # This should load model.pkl
        st.session_state.df = load_data()
        st.session_state.preprocessed_df, _, _ = preprocess_data(st.session_state.df)
        st.session_state.data_loaded = True

# Display Temperature and Humidity sliders side by side with an expanded temperature range
cols = st.columns(2)
with cols[0]:
    temp = st.slider("Temperature (°C)", min_value=-50.0, max_value=45.0, value=20.0, step=0.5)
with cols[1]:
    humidity = st.slider("Humidity (%)", min_value=20.0, max_value=90.0, value=50.0, step=1.0)

# Zone filter section (placed below the temperature and humidity sliders)
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

if st.button("Run Optimization", type="primary"):
    with st.spinner("Processing..."):
        # Use cached data instead of reloading
        result_df = assign_workers_to_regions(zone_df, st.session_state.preprocessed_df, temp, humidity)
        
        # Process the result dataframe to build formatted results based on individual worker details
        formatted_results = []
        for _, row in result_df.iterrows():
            zone = row['Zone']
            processed_quantity = row['Processed_Quantity']
            team_size = row['Team Size']
            team_etc = row['EstimatedTimeToPickTheQuantity']
            team_productivity = row['Team Productivity']
            worker_details = row['WorkerDetails']  # list of dicts with keys: worker_id, Individual ETC, Individual Productivity
            
            for detail in worker_details:
                formatted_results.append({
                    "Zone": f"Zone {zone}",
                    "Proposed Team": detail["worker_id"],
                    "Processed Quantity": processed_quantity,
                    "Team Size": team_size,
                    "Individual ETC": detail["Individual ETC"],
                    "Individual Productivity": format(round(float(str(detail["Individual Productivity"]).replace(" items/hr", ""))), '.0f'),
                    "Team ETC": team_etc,
                    "Team Productivity": format(round(float(str(team_productivity).replace(" items/hr", ""))), '.0f')
                })
        
        formatted_df = pd.DataFrame(formatted_results)
        
        # Create HTML table with merged cells for zones
        st.subheader("Worker Assignments")
        html_table = """
        <div style="overflow-x: auto;">
        <table style="width:100%; border-collapse: collapse; margin-top: 20px;">
            <thead>
                <tr style="background-color: #4e8cff; color: white;">
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
                    html_table += f"<td style='padding: 10px; border: 1px solid #ddd;'>{row['Individual Productivity']}</td>"
                    html_table += f"<td style='padding: 10px; border: 1px solid #ddd;' rowspan='{rows}'>{row['Team ETC']}</td>"
                    html_table += f"<td style='padding: 10px; border: 1px solid #ddd;' rowspan='{rows}'>{row['Team Productivity']}</td>"
                else:
                    html_table += f"<td style='padding: 10px; border: 1px solid #ddd;'>{row['Proposed Team']}</td>"
                    html_table += f"<td style='padding: 10px; border: 1px solid #ddd;'>{row['Individual ETC']}</td>"
                    html_table += f"<td style='padding: 10px; border: 1px solid #ddd;'>{row['Individual Productivity']}</td>"
                html_table += "</tr>"
                first_row = False
        
        html_table += """
            </tbody>
        </table>
        </div>
        """
        st.markdown(html_table, unsafe_allow_html=True)

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

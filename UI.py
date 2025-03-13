import streamlit as st
import pandas as pd
import numpy as np
import pickle
from itertools import combinations
import os
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

# Import functions from your model.py file
from model import load_data, preprocess_data, train_model, load_model

st.set_page_config(page_title="Warehouse Worker Assignment Tool", layout="wide")

# Custom CSS to improve styling with professional colors
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton button {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
    }
    /* Professional color scheme for metrics */
    .metric-card-1 {
        background-color: #E1F5FE;
        color: #333;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
    }
    .metric-card-2 {
        background-color: #E8F5E9;
        color: #333;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
    }
    .metric-card-3 {
        background-color: #FFF8E1;
        color: #333;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
    }
    .metric-card-4 {
        background-color: #F3E5F5;
        color: #333;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 16px;
        opacity: 0.8;
    }
    .card {
        background-color: #f2f4f5;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    h1 {
        color: #0066cc;
        font-weight: bold;
    }
    h2, h3 {
        color: #0066cc;
    }
    .insight-section {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        border-left: 4px solid #0066cc;
    }
    .optimization-card {
        background-color: #f5f7fa;
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 16px;
    }
    /* Hide dataframe index */
    .row_heading, .blank {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Helper: Improved safe_qcut function
# ---------------------------
def safe_qcut(series, q, default_labels):
    """
    Attempts to bin a series into q quantile-based categories.
    If there are fewer unique values than q, uses pd.cut with linearly spaced bins.
    """
    if series.nunique() < q:
        try:
            bins = np.linspace(series.min(), series.max(), q+1)
            if len(np.unique(bins)) < 2:
                return pd.Series([default_labels[q//2]] * len(series))
            return pd.cut(series, bins=bins, labels=default_labels, include_lowest=True)
        except Exception as e:
            return pd.Series([default_labels[q//2]] * len(series))
    try:
        return pd.qcut(series, q=q, labels=default_labels, duplicates="drop")
    except Exception as e:
        return pd.Series([default_labels[q//2]] * len(series))

# ---------------------------
# Heatmap color function for Productivity Level
# ---------------------------
def color_productivity(val):
    if val == "Excellent":
        return 'background-color: #1e88e5; color: white'
    elif val == "High":
        return 'background-color: #42a5f5; color: white'
    elif val == "Medium":
        return 'background-color: #90caf9; color: black'
    else:
        return 'background-color: #e3f2fd; color: black'

# ---------------------------
# Load model once at startup
# ---------------------------
@st.cache_resource
def load_model_singleton():
    model_filename = "model.pkl"
    if os.path.exists(model_filename):
        return load_model(model_filename)
    else:
        df = load_data()
        df_filtered = preprocess_data(df)
        train_model(df_filtered)
        return load_model(model_filename)

# ---------------------------
# Optimized functions with updated extrapolation logic and variability enhancements
# ---------------------------
def get_worker_features(worker, region, temp, humidity, hour, temp_hum_interaction, features, df_cache):
    """
    Retrieves worker features, always using the current input for temperature-related fields.
    Additionally, if a historical temperature is available, it is stored separately.
    """
    df_region = df_cache.get((region, worker), None)
    if df_region is None or df_region.empty:
        return None
    worker_features = {}
    # Save historical temperature if available
    if "temperature" in df_region.columns:
        worker_features["historical_temperature"] = df_region["temperature"].iloc[0]
    for feature in features:
        if feature in ["temperature", "humidity", "hour", "temp_hum_interaction"]:
            if feature == "temperature":
                worker_features[feature] = temp  # override with current temperature
            elif feature == "humidity":
                worker_features[feature] = humidity
            elif feature == "hour":
                worker_features[feature] = hour
            elif feature == "temp_hum_interaction":
                worker_features[feature] = temp_hum_interaction
        else:
            if feature in df_region.columns:
                worker_features[feature] = df_region[feature].iloc[0]
            else:
                worker_features[feature] = 0
    return worker_features

def evaluate_worker_combinations_batch(worker_combinations, region, df_cache, temp, humidity, hour, 
                                       temp_hum_interaction, model, scaler, features):
    """
    Evaluate each combination of workers.
    After computing the base prediction, adjust the score based on the difference between
    the current temperature and the workers’ historical temperatures. A small random noise is added
    to help differentiate similar combinations.
    """
    results = []
    k_adjust = 0.2  # Increased adjustment coefficient per degree difference
    for combo in worker_combinations:
        combo_features = []
        historical_temps = []
        for worker in combo:
            worker_features = get_worker_features(worker, region, temp, humidity, hour, 
                                                  temp_hum_interaction, features, df_cache)
            if worker_features:
                combo_features.append(worker_features)
                if "historical_temperature" in worker_features:
                    historical_temps.append(worker_features["historical_temperature"])
        if not combo_features:
            continue
        avg_features = {}
        for feature in features:
            avg_features[feature] = sum(w.get(feature, 0) for w in combo_features) / len(combo_features)
        features_df = pd.DataFrame([list(avg_features.values())], columns=features)
        features_scaled = scaler.transform(features_df)
        # Obtain the base prediction score from the model
        score = model.predict(features_scaled)[0]
        # Adjust based on temperature difference if historical data exists
        if historical_temps:
            avg_hist_temp = sum(historical_temps) / len(historical_temps)
            temp_diff = abs(temp - avg_hist_temp)
            adjustment_factor = max(0.5, 1 - k_adjust * temp_diff)
            score = score * adjustment_factor
        # Add a small random noise to create variation among similar teams
        noise = np.random.normal(scale=0.1)
        score = score + noise
        results.append((combo, score))
    return results

@st.cache_data(ttl=300)
def recommend_worker_combinations(df, temp, humidity, unique_hours, combination_size=2, region_filter="All"):
    """
    Recommend optimal worker combinations and flag if extrapolation is occurring.
    """
    start_time = time.time()
    model_data = load_model_singleton()
    model, scaler, features = model_data["model"], model_data["scaler"], model_data["features"]
    
    # Determine training temperature range
    temp_min = df["temperature"].min()
    temp_max = df["temperature"].max()
    # Flag if current temperature is outside the training range
    extrapolation_needed = (temp < temp_min) or (temp > temp_max)
    
    region_assignments = {}
    current_hour = pd.Timestamp.now().hour
    selected_hour = current_hour if current_hour in unique_hours else (unique_hours[0] if unique_hours else 12)
    
    temp_hum_interaction = temp * humidity
    df_cache = {}
    for _, row in df.iterrows():
        region = row["region"]
        worker = row["resource"]
        key = (region, worker)
        if key not in df_cache:
            df_cache[key] = df[(df["region"] == region) & (df["resource"] == worker)]
    
    # Build dictionary of workers per region
    region_workers = {}
    for region in df["region"].unique():
        region_workers[region] = set(df[df["region"] == region]["resource"].unique())
    
    regions_sorted = sorted(region_workers.keys(), key=lambda r: len(region_workers[r]), reverse=True)
    max_threads = min(8, os.cpu_count() or 4)
    batch_size = 50

    global_assigned = set() if region_filter == "All" else None

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for region in regions_sorted:
            if region_filter == "All":
                eligible_workers = [w for w in list(region_workers[region]) if w not in global_assigned]
            else:
                eligible_workers = list(region_workers[region])
            if len(eligible_workers) < combination_size:
                continue
            worker_combinations = list(combinations(eligible_workers, combination_size))
            for i in range(0, len(worker_combinations), batch_size):
                batch = worker_combinations[i:i+batch_size]
                future = executor.submit(
                    evaluate_worker_combinations_batch,
                    batch, 
                    region,
                    df_cache,
                    temp,
                    humidity,
                    selected_hour,
                    temp_hum_interaction,
                    model,
                    scaler,
                    features
                )
                futures.append((region, future))
        region_results = {}
        for region, future in futures:
            region_results.setdefault(region, [])
            batch_results = future.result()
            if batch_results:
                region_results[region].extend(batch_results)
        for region, results in region_results.items():
            if not results:
                continue
            sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
            if region_filter == "All":
                # Enforce global uniqueness across regions when needed.
                top_combinations = sorted_results if len(sorted_results) < 5 else sorted_results[:5]
                if top_combinations:
                    best_combo = top_combinations[0][0]
                    global_assigned.update(best_combo)
                region_assignments[region] = top_combinations
            else:
                top_combinations = sorted_results if len(sorted_results) < 5 else sorted_results[:5]
                region_assignments[region] = top_combinations
    processing_time = time.time() - start_time
    return {
        "assignments": region_assignments,
        "processing_time": processing_time,
        "extrapolation_info": {
            "needed": extrapolation_needed,
            "temp_min": temp_min,
            "temp_max": temp_max,
            "current_temp": temp
        }
    }

def calculate_kpis(best_assignments, df_filtered, processing_time, extrapolation_info):
    """
    Calculate KPIs using the best assignments based on productivity scores.
    """
    total_workers = sum(best_assignments["Team Size"])
    total_regions = len(best_assignments)
    avg_productivity = best_assignments["Productivity Score"].mean() if not best_assignments["Productivity Score"].empty else 0
    total_regions_count = len(df_filtered["region"].unique())
    coverage_ratio = (total_regions / total_regions_count) * 100 if total_regions_count > 0 else 0
    total_available_workers = len(df_filtered["resource"].unique())
    worker_utilization = (total_workers / total_available_workers) * 100 if total_available_workers > 0 else 0
    return {
        "avg_productivity": float(round(avg_productivity, 2)),
        "coverage_ratio": float(round(coverage_ratio, 2)),
        "worker_utilization": float(round(worker_utilization, 2)),
        "processing_time": float(round(processing_time, 2)),
        "extrapolation_needed": bool(extrapolation_info["needed"])
    }

def display_results(recommendations, temp, humidity, df_filtered):
    """Display worker assignment results with KPIs and insights"""
    region_assignments = recommendations["assignments"]
    processing_time = recommendations["processing_time"]
    extrapolation_info = recommendations["extrapolation_info"]
    
    st.header("Optimal Worker Assignments")
    
    # Build the recommended assignment table with columns:
    # Region, Team, Productivity Score.
    results_data = []
    for region, combinations in region_assignments.items():
        for idx, (workers, score) in enumerate(combinations):
            worker_names = ", ".join(workers)
            productivity_score = round(score / len(workers), 2)
            results_data.append({
                "Region": region,
                "Team": worker_names,
                "Productivity Score": productivity_score
            })
    
    results_df = pd.DataFrame(results_data).reset_index(drop=True)
    if results_df.empty:
        st.error("No valid assignments could be generated. Try adjusting your parameters.")
        return

    # Add Productivity Level to the table (using safe_qcut)
    results_df["Productivity Level"] = safe_qcut(
        results_df["Productivity Score"],
        q=4,
        default_labels=["Low", "Medium", "High", "Excellent"]
    )
    
    # Create a styler to hide the dataframe index and apply heatmap coloring to Productivity Level.
    styled_display_df = results_df.style.applymap(color_productivity, subset=["Productivity Level"]).set_table_styles(
        [{'selector': 'th.row_heading', 'props': [('display', 'none')]},
         {'selector': '.blank', 'props': [('display', 'none')]}]
    )
    
    # Create tabs for organization
    tab1, tab2 = st.tabs(["Optimal Assignments", "Performance Analytics"])
    
    # For KPI calculations, select the top-ranked assignment per region.
    best_assignments_df = results_df.groupby("Region", as_index=False).first()
    # Compute Team Size from the Team column (number of workers per team)
    best_assignments_df["Team Size"] = best_assignments_df["Team"].apply(
        lambda x: len([w.strip() for w in x.split(",") if w.strip()])
    )
    # Compute union of worker names across regions
    all_workers = set()
    for team in best_assignments_df["Team"]:
        all_workers.update([w.strip() for w in team.split(",") if w.strip()])
    total_workers_assigned = len(all_workers)
    
    kpis = calculate_kpis(best_assignments_df, df_filtered, processing_time, extrapolation_info)
    
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # Display four KPI cards: Temperature, Humidity, Total Workers, and Avg Productivity.
        cols = st.columns(4)
        
        cols[0].markdown(f"""
        <div class="metric-card-1">
            <div class="metric-label">Temperature</div>
            <div class="metric-value">{temp:.2f}°C</div>
            <div>Current condition</div>
        </div>
        """, unsafe_allow_html=True)
        
        cols[1].markdown(f"""
        <div class="metric-card-2">
            <div class="metric-label">Humidity</div>
            <div class="metric-value">{humidity:.2f}%</div>
            <div>Current condition</div>
        </div>
        """, unsafe_allow_html=True)
        
        cols[2].markdown(f"""
        <div class="metric-card-3">
            <div class="metric-label">Total Workers</div>
            <div class="metric-value">{total_workers_assigned}</div>
            <div>Assigned</div>
        </div>
        """, unsafe_allow_html=True)
        
        cols[3].markdown(f"""
        <div class="metric-card-4">
            <div class="metric-label">Avg Productivity</div>
            <div class="metric-value">{kpis['avg_productivity']:.2f}</div>
            <div>Score</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Recommended Assignments")
        st.markdown(styled_display_df.to_html(), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="insight-section">', unsafe_allow_html=True)
        st.subheader("Statistical Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Productivity Statistics")
            max_prod = round(best_assignments_df['Productivity Score'].max(), 2)
            min_prod = round(best_assignments_df['Productivity Score'].min(), 2)
            avg_prod = round(best_assignments_df['Productivity Score'].mean(), 2)
            std_dev = round(best_assignments_df['Productivity Score'].std(ddof=0), 2) if len(best_assignments_df) > 1 else 0
            st.markdown(f"""
            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;">
                <div style="flex: 1; min-width: 120px; padding: 12px; background: #f0f7ff; border-radius: 6px; border-left: 4px solid #1e88e5; text-align: center;">
                    <div style="font-size: 14px; color: #555;">Maximum</div>
                    <div style="font-size: 22px; font-weight: bold;">{max_prod:.2f}</div>
                </div>
                <div style="flex: 1; min-width: 120px; padding: 12px; background: #f0f7ff; border-radius: 6px; border-left: 4px solid #1e88e5; text-align: center;">
                    <div style="font-size: 14px; color: #555;">Minimum</div>
                    <div style="font-size: 22px; font-weight: bold;">{min_prod:.2f}</div>
                </div>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                <div style="flex: 1; min-width: 120px; padding: 12px; background: #f0f7ff; border-radius: 6px; border-left: 4px solid #1e88e5; text-align: center;">
                    <div style="font-size: 14px; color: #555;">Average</div>
                    <div style="font-size: 22px; font-weight: bold;">{avg_prod:.2f}</div>
                </div>
                <div style="flex: 1; min-width: 120px; padding: 12px; background: #f0f7ff; border-left: 4px solid #1e88e5; text-align: center;">
                    <div style="font-size: 14px; color: #555;">Std Dev</div>
                    <div style="font-size: 22px; font-weight: bold;">{std_dev:.2f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("### Efficiency Statistics")
            high_eff = round(best_assignments_df['Productivity Score'].max(), 2)
            low_eff = round(best_assignments_df['Productivity Score'].min(), 2)
            avg_eff = round(best_assignments_df['Productivity Score'].mean(), 2)
            st.markdown(f"""
            <div style="display: flex; gap: 10px; margin-bottom: 10px;">
                <div style="flex: 1; background: #fff8e1; padding: 12px; border-radius: 6px; text-align: center;">
                    <div style="font-size: 14px; color: #555;">Highest</div>
                    <div style="font-size: 22px; font-weight: bold;">{high_eff:.2f}</div>
                </div>
                <div style="flex: 1; background: #fff8e1; padding: 12px; border-radius: 6px; text-align: center;">
                    <div style="font-size: 14px; color: #555;">Lowest</div>
                    <div style="font-size: 22px; font-weight: bold;">{low_eff:.2f}</div>
                </div>
            </div>
            <div style="background: #fff8e1; padding: 12px; border-radius: 6px; text-align: center;">
                <div style="font-size: 14px; color: #555;">Average Efficiency</div>
                <div style="font-size: 22px; font-weight: bold;">{avg_eff:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <footer>
      <hr style="border: 0; border-top: 1px solid #eee;">
      <p>&copy; 2025 Warehouse Worker Assignment Optimizer</p>
    </footer>
    """, unsafe_allow_html=True)

def main():
    st.title("Warehouse Worker Assignment Optimizer")
    
    # Input form (UI remains unchanged except for updated temperature range)
    with st.form("optimization_form"):
        col1, col2 = st.columns([1, 1])
        with col1:
            temp_input = st.number_input("Temperature (°C)", min_value=-20.0, max_value=40.0, value=22.0, step=0.5,
                                        help="Enter temperature in °C. The application extrapolates predictions for temperatures outside the training data range, including negative values.", format="%.2f")
        with col2:
            humidity_input = st.number_input("Humidity (%)", min_value=20.0, max_value=95.0, value=60.0, step=1.0, format="%.2f")
        
        col1, col2 = st.columns(2)
        with col1:
            combination_size = st.selectbox("Workers per Team", options=[1, 2, 3, 4], index=1)
        with col2:
            region_filter = st.selectbox("Region", options=["All", "1", "2", "3", "4", "5", "6"], index=0)
        
        submit_button = st.form_submit_button("Generate Optimal Assignments")
    
    if submit_button:
        try:
            with st.spinner("Calculating optimal worker assignments..."):
                progress_bar = st.progress(0)
                progress_bar.progress(10)
                df = load_data()
                # Filter data based on region selection if not "All"
                if region_filter != "All":
                    df = df[df["region"].astype(str) == region_filter]
                progress_bar.progress(30)
                df_filtered = preprocess_data(df)
                unique_hours = sorted(df_filtered["hour"].dropna().unique())
                progress_bar.progress(50)
                recommendations = recommend_worker_combinations(
                    df_filtered, temp_input, humidity_input, unique_hours, 
                    combination_size=combination_size, region_filter=region_filter
                )
                progress_bar.progress(100)
                if recommendations["assignments"]:
                    display_results(recommendations, temp_input, humidity_input, df_filtered)
                else:
                    st.warning("No recommendations could be generated. Try adjusting your parameters or check your dataset.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            # st.info("Make sure your 'model.py' file is in the same directory and contains the necessary functions.")

if __name__ == "__main__":
    main()

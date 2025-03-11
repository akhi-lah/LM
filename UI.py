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
import os
import subprocess
import sys

def install_dependencies():
    try:
        import streamlit
        import xgboost
    except ImportError:
        print("Installing missing dependencies...")
        subprocess.call([sys.executable, "-m", "pip", "install", "streamlit", "xgboost"])


install_dependencies()
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
</style>
""", unsafe_allow_html=True)

# Load model once at startup to avoid repeated loading
@st.cache_resource
def load_model_singleton():
    model_filename = "model.pkl"
    if os.path.exists(model_filename):
        return load_model(model_filename)
    else:
        # First-time initialization
        df = load_data()
        df_filtered = preprocess_data(df)
        train_model(df_filtered)
        return load_model(model_filename)

# Get all available regions from the data
@st.cache_data
def get_all_regions(df):
    regions = sorted(df["region"].unique().tolist())
    return ["All"] + regions

# Optimized functions with LRU cache to improve performance

# Replace the cache-related functions with these improved versions
def get_worker_features(worker, region, temp, humidity, hour, temp_hum_interaction, features, df_cache):
    """Get worker features without caching for improved reliability"""
    # Get worker data from cache
    df_region = df_cache.get((region, worker), None)
    if df_region is None or df_region.empty:
        return None
    
    # Create worker features
    worker_features = {}
    for feature in features:
        if feature in df_region.columns:
            worker_features[feature] = df_region[feature].iloc[0]
        else:
            # Handle special case features
            if feature == "temperature":
                worker_features[feature] = temp
            elif feature == "humidity":
                worker_features[feature] = humidity
            elif feature == "hour":
                worker_features[feature] = hour
            elif feature == "temp_hum_interaction":
                worker_features[feature] = temp_hum_interaction
            else:
                worker_features[feature] = 0  # Default value
    
    return worker_features

def evaluate_worker_combinations_batch(worker_combinations, region, df_cache, temp, humidity, hour, 
                                    temp_hum_interaction, model, scaler, features):
    results = []
    
    for combo in worker_combinations:
        combo_features = []
        for worker in combo:
            # Get worker features
            worker_features = get_worker_features(worker, region, temp, humidity, hour, 
                                              temp_hum_interaction, features, df_cache)
            if worker_features:
                combo_features.append(worker_features)
        
        if not combo_features:
            continue
            
        # Calculate average features for the combination
        avg_features = {}
        for feature in features:
            avg_features[feature] = sum(w.get(feature, 0) for w in combo_features) / len(combo_features)
        
        # Convert to DataFrame and scale
        features_df = pd.DataFrame([list(avg_features.values())], columns=features)
        features_scaled = scaler.transform(features_df)
        
        # Predict performance
        score = model.predict(features_scaled)[0]
        results.append((combo, score))
    
    return results

# Highly optimized recommendation function with improved caching and parallel processing
@st.cache_data(ttl=300)  # Cache for 5 minutes
def recommend_worker_combinations(df, temp, humidity, unique_hours, selected_region="All", combination_size=2, top_n=5):
    """
    Recommend optimal worker combinations with significant performance improvements:
    - Precomputed caches for worker data
    - Parallel processing with optimized batching
    - Worker feature caching with LRU cache
    - Support for filtering by selected region
    """
    start_time = time.time()
    
    # Load model just once
    model_data = load_model_singleton()
    model, scaler, features = model_data["model"], model_data["scaler"], model_data["features"]
    
    # Get model's training range for temperature (for extrapolation notice)
    temp_min = df["temperature"].min()
    temp_max = df["temperature"].max()
    extrapolation_needed = temp < temp_min or temp > temp_max
    
    region_assignments = {}
    
    # Precompute current hour or select appropriate hour from dataset
    current_hour = pd.Timestamp.now().hour
    if current_hour in unique_hours:
        selected_hour = current_hour
    else:
        selected_hour = unique_hours[0] if len(unique_hours) > 0 else 12
    
    # Precompute temp-humidity interaction
    temp_hum_interaction = temp * humidity
    
    # Precompute worker data cache for all regions (major optimization)
    df_cache = {}
    for _, row in df.iterrows():
        region = row["region"]
        worker = row["resource"]
        key = (region, worker)
        if key not in df_cache:
            df_cache[key] = df[(df["region"] == region) & (df["resource"] == worker)]
    
    # Precompute region workers (avoids repeated filtering)
    region_workers = {}
    for region in df["region"].unique():
        region_workers[region] = set(df[df["region"] == region]["resource"].unique())
    
    # Filter regions based on selection
    if selected_region != "All":
        regions_to_process = [selected_region] if selected_region in region_workers else []
    else:
        regions_to_process = sorted(region_workers.keys())
    
    # Determine optimal parallel processing setup
    max_threads = min(8, os.cpu_count() or 4)  # Cap at 8 threads or available CPUs
    batch_size = 50  # Process combinations in batches for better efficiency
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        
        for region in regions_to_process:
            eligible_workers = list(region_workers[region])
            
            if len(eligible_workers) < combination_size:
                continue
                
            # Generate combinations and split into batches
            worker_combinations = list(combinations(eligible_workers, combination_size))
            
            # Process in batches for better performance
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
        
        # Collect results by region
        region_results = {}
        for region, future in futures:
            if region not in region_results:
                region_results[region] = []
            
            batch_results = future.result()
            if batch_results:
                region_results[region].extend(batch_results)
        
        # Process results for each region
        for region, results in region_results.items():
            if not results:
                continue
                
            # Find top combinations for this region
            # If a specific region is selected, get top 5, otherwise get just the top 1
            combinations_to_keep = 5 if selected_region != "All" else 1
            top_combinations = sorted(results, key=lambda x: x[1], reverse=True)[:combinations_to_keep]
            if top_combinations:
                region_assignments[region] = top_combinations
    
    # Calculate processing time
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

# Function to calculate meaningful KPIs
def calculate_kpis(best_assignments, df_filtered, processing_time, extrapolation_info):
    # Calculate productivity metrics
    total_workers = sum(best_assignments["Team Size"])
    total_regions = len(best_assignments)
    avg_prediction = best_assignments["Predicted Quantity"].mean()
    total_prediction = best_assignments["Predicted Quantity"].sum()
    
    # Worker efficiency (predicted items per worker)
    worker_efficiency = total_prediction / total_workers if total_workers > 0 else 0
    
    # Warehouse coverage ratio (percentage of regions covered)
    total_regions_count = len(df_filtered["region"].unique())
    coverage_ratio = (total_regions / total_regions_count) * 100 if total_regions_count > 0 else 0
    
    # Worker utilization (percentage of total available workers assigned)
    total_available_workers = len(df_filtered["resource"].unique())
    worker_utilization = (total_workers / total_available_workers) * 100 if total_available_workers > 0 else 0
    
    # Estimated completion time (based on productivity)
    avg_items_per_hour = worker_efficiency  # Simplified assumption
    warehouse_workload = total_regions * 1000  # Simplified estimate of workload
    est_completion_time = warehouse_workload / total_prediction if total_prediction > 0 else 0
    
    # Environmental optimization score (0-100)
    temp = extrapolation_info["current_temp"]
    temp_min = extrapolation_info["temp_min"]
    temp_max = extrapolation_info["temp_max"]
    
    # Higher score if temperature is in optimal range
    if 19 <= temp <= 25:
        env_score = 90
    elif temp < temp_min or temp > temp_max:
        # Lower score for extrapolated temperatures
        env_score = 70
    else:
        env_score = 80
    
    # Combine into a dictionary with JSON-serializable values
    return {
        "worker_efficiency": float(round(worker_efficiency, 1)),
        "coverage_ratio": float(round(coverage_ratio, 1)),
        "worker_utilization": float(round(worker_utilization, 1)),
        "est_completion_hours": float(round(est_completion_time, 1)),
        "env_optimization_score": int(env_score),
        "processing_time": float(round(processing_time, 3)),
        "extrapolation_needed": bool(extrapolation_info["needed"])
    }

def display_results(recommendations, temp, humidity, df_filtered, selected_region):
    """Display worker assignment results with KPIs and insights"""
    # Extract data from recommendations
    region_assignments = recommendations["assignments"]
    processing_time = recommendations["processing_time"]
    extrapolation_info = recommendations["extrapolation_info"]
    
    # Display extrapolation warning if needed
    # if extrapolation_info["needed"]:
    #     st.warning(f"Warning: Extrapolating predictions for temperature {temp}°C. Model was trained on temperatures between {extrapolation_info['temp_min']}°C and {extrapolation_info['temp_max']}°C. Predictions may be less reliable.")
    
    st.header("Optimal Worker Combination")
    
    # Prepare data for the results table
    results_data = []
    for region, combinations in region_assignments.items():
        for idx, (workers, score) in enumerate(combinations):
            worker_names = ", ".join(workers)
            # Calculate a productivity score (score per team member)
            productivity_score = round(score / len(workers), 1)
            results_data.append({
                "Region": region,
                "Team": worker_names,
                "Predicted Quantity": int(score),
                "Productivity Score": productivity_score,
                "Rank": idx + 1,
                "Team Size": len(workers)
            })
    
    results_df = pd.DataFrame(results_data)
    
    if results_df.empty:
        st.error("No valid assignments could be generated. Try adjusting your parameters.")
        return
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["Optimal Assignments", "Performance Analytics"])
    
    # Calculate meaningful KPIs
    best_assignments = results_df.copy()
    if selected_region == "All":
        # For "All" regions, we only want rank 1 for each region
        best_assignments = results_df[results_df["Rank"] == 1].copy()
    # Keep all ranks for a specific region selection
    best_assignments = best_assignments.drop("Rank", axis=1)
    kpis = calculate_kpis(best_assignments, df_filtered, processing_time, extrapolation_info)
    
    with tab1:
        # Display environment KPI cards at the top
        st.markdown('<div class="card">', unsafe_allow_html=True)
        cols = st.columns(4)
        
        # Temperature KPI - Limit to 2 decimal places
        cols[0].markdown(f"""
        <div class="metric-card-1">
            <div class="metric-label">Temperature</div>
            <div class="metric-value">{temp:.2f}°C</div>
            <div>Current condition</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Humidity KPI - Limit to 2 decimal places
        cols[1].markdown(f"""
        <div class="metric-card-2">
            <div class="metric-label">Humidity</div>
            <div class="metric-value">{humidity:.2f}%</div>
            <div>Current condition</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Total Workers KPI
        cols[2].markdown(f"""
        <div class="metric-card-3">
            <div class="metric-label">Total Workers</div>
            <div class="metric-value">{int(sum(best_assignments['Team Size']))}</div>
            <div>Assigned</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Total Production KPI
        cols[3].markdown(f"""
        <div class="metric-card-4">
            <div class="metric-label">Total Production</div>
            <div class="metric-value">{int(sum(best_assignments['Predicted Quantity']))}</div>
            <div>Expected units</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display the assignments table with modified columns
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Recommended Assignments")
        
        # Add information about the region filter
        if selected_region != "All":
            st.info(f"Showing top 5 team combinations for region: {selected_region}")
        else:
            st.info("Showing best team combination for each region")
        
        # Display only selected columns and sort by productivity score
        display_df = best_assignments[["Region", "Team", "Predicted Quantity", "Productivity Score"]].sort_values("Productivity Score", ascending=False)
        st.dataframe(display_df, use_container_width=True)
        
        # Add a heatmap visualization of productivity by region
        st.subheader("Regional Productivity Heatmap")
        
        # Create a simple heatmap as a visualization
        heatmap_data = display_df.copy()
        # heatmap_data["Productivity Level"] = pd.qcut(heatmap_data["Productivity Score"], 
        #                                             q=4, 
        #                                             labels=["Low", "Medium", "High", "Excellent"])
        try:
          heatmap_data["Productivity Level"] = pd.qcut(heatmap_data["Productivity Score"], 
                                                    q=4, 
                                                    labels=["Low", "Medium", "High", "Excellent"])
        except ValueError:
        # Fallback if we can't create 4 categories: use equal-width bins instead
            heatmap_data["Productivity Level"] = pd.cut(heatmap_data["Productivity Score"],
                                                  bins=4,
                                                  labels=["Low", "Medium", "High", "Excellent"])
        
        # Use Streamlit's native table with styling
        def color_productivity(val):
            if val == "Excellent":
                return 'background-color: #1e88e5; color: white'
            elif val == "High":
                return 'background-color: #42a5f5; color: white'
            elif val == "Medium":
                return 'background-color: #90caf9; color: black'
            else:
                return 'background-color: #e3f2fd; color: black'
                
        styled_heatmap = heatmap_data[["Region", "Productivity Score", "Productivity Level"]].style.applymap(
            color_productivity, subset=["Productivity Level"]
        )
        
        st.dataframe(styled_heatmap, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        # Statistical Analysis Section
        st.markdown('<div class="insight-section">', unsafe_allow_html=True)
        st.subheader("Statistical Analysis")

        # Create visual cards instead of tables
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Productivity Statistics")
            
            # Calculate key metrics
            max_prod = int(best_assignments['Predicted Quantity'].max())
            min_prod = int(best_assignments['Predicted Quantity'].min())
            avg_prod = int(best_assignments['Predicted Quantity'].mean())
            std_dev = round(best_assignments['Predicted Quantity'].std(), 1)
            total_prod = int(best_assignments['Predicted Quantity'].sum())
            
            # Display as visual cards
            st.markdown(f"""
            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;">
                <div style="flex: 1; min-width: 120px; padding: 12px; background: #f0f7ff; border-radius: 6px; border-left: 4px solid #1e88e5;">
                    <div style="font-size: 14px; color: #555;">Maximum</div>
                    <div style="font-size: 22px; font-weight: bold;">{max_prod}</div>
                    <div style="font-size: 12px; color: #777;">units</div>
                </div>
                <div style="flex: 1; min-width: 120px; padding: 12px; background: #f0f7ff; border-radius: 6px; border-left: 4px solid #1e88e5;">
                    <div style="font-size: 14px; color: #555;">Minimum</div>
                    <div style="font-size: 22px; font-weight: bold;">{min_prod}</div>
                    <div style="font-size: 12px; color: #777;">units</div>
                </div>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;">
                <div style="flex: 1; min-width: 120px; padding: 12px; background: #f0f7ff; border-radius: 6px; border-left: 4px solid #1e88e5;">
                    <div style="font-size: 14px; color: #555;">Average</div>
                    <div style="font-size: 22px; font-weight: bold;">{avg_prod}</div>
                    <div style="font-size: 12px; color: #777;">units</div>
                </div>
                <div style="flex: 1; min-width: 120px; padding: 12px; background: #f0f7ff; border-radius: 6px; border-left: 4px solid #1e88e5;">
                    <div style="font-size: 14px; color: #555;">Standard Dev.</div>
                    <div style="font-size: 22px; font-weight: bold;">{std_dev}</div>
                    <div style="font-size: 12px; color: #777;">units</div>
                </div>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                <div style="flex: 1; padding: 12px; background: #e8f5e9; border-radius: 6px; border-left: 4px solid #43a047;">
                    <div style="font-size: 14px; color: #555;">Total Production</div>
                    <div style="font-size: 24px; font-weight: bold;">{total_prod}</div>
                    <div style="font-size: 12px; color: #777;">units</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("### Efficiency Statistics")
            
            # Calculate efficiency metrics
            high_eff = round(best_assignments['Productivity Score'].max(), 1)
            low_eff = round(best_assignments['Productivity Score'].min(), 1)
            avg_eff = round(best_assignments['Productivity Score'].mean(), 1)
            top_quartile = round(best_assignments['Productivity Score'].quantile(0.75), 1)
            bottom_quartile = round(best_assignments['Productivity Score'].quantile(0.25), 1)
            
            # Display as visual cards
            st.markdown(f"""
            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;">
                <div style="flex: 1; min-width: 120px; padding: 12px; background: #fff8e1; border-radius: 6px; border-left: 4px solid #ffa000;">
                    <div style="font-size: 14px; color: #555;">Highest</div>
                    <div style="font-size: 22px; font-weight: bold;">{high_eff}</div>
                    <div style="font-size: 12px; color: #777;">units/worker</div>
                </div>
                <div style="flex: 1; min-width: 120px; padding: 12px; background: #fff8e1; border-radius: 6px; border-left: 4px solid #ffa000;">
                    <div style="font-size: 14px; color: #555;">Lowest</div>
                    <div style="font-size: 22px; font-weight: bold;">{low_eff}</div>
                    <div style="font-size: 12px; color: #777;">units/worker</div>
                </div>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;">
                <div style="flex: 1; min-width: 120px; padding: 12px; background: #fff8e1; border-radius: 6px; border-left: 4px solid #ffa000;">
                    <div style="font-size: 14px; color: #555;">Top 25%</div>
                    <div style="font-size: 22px; font-weight: bold;">≥ {top_quartile}</div>
                    <div style="font-size: 12px; color: #777;">units/worker</div>
                </div>
                <div style="flex: 1; min-width: 120px; padding: 12px; background: #fff8e1; border-radius: 6px; border-left: 4px solid #ffa000;">
                    <div style="font-size: 14px; color: #555;">Bottom 25%</div>
                    <div style="font-size: 22px; font-weight: bold;">≤ {bottom_quartile}</div>
                    <div style="font-size: 12px; color: #777;">units/worker</div>
                </div>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                <div style="flex: 1; padding: 12px; background: #f3e5f5; border-radius: 6px; border-left: 4px solid #8e24aa;">
                    <div style="font-size: 14px; color: #555;">Average Efficiency</div>
                    <div style="font-size: 24px; font-weight: bold;">{avg_eff}</div>
                    <div style="font-size: 12px; color: #777;">units/worker</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Replace the Performance Distribution Analysis with a cleaner visualization
        st.markdown('<div class="insight-section">', unsafe_allow_html=True)
        st.subheader("Performance Distribution Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Calculate quartiles for worker efficiency
            productivity_quartiles = best_assignments["Productivity Score"].quantile([0.25, 0.5, 0.75]).to_dict()
            
            # Create distribution data
            bottom_regions = best_assignments[best_assignments["Productivity Score"] <= productivity_quartiles[0.25]]
            lower_mid_regions = best_assignments[(best_assignments["Productivity Score"] > productivity_quartiles[0.25]) & 
                                            (best_assignments["Productivity Score"] <= productivity_quartiles[0.5])]
            upper_mid_regions = best_assignments[(best_assignments["Productivity Score"] > productivity_quartiles[0.5]) & 
                                            (best_assignments["Productivity Score"] <= productivity_quartiles[0.75])]
            top_regions = best_assignments[best_assignments["Productivity Score"] > productivity_quartiles[0.75]]
            
            # Visual distribution chart with tooltips
            st.markdown("### Productivity Distribution")
            
            # Create a visual representation of distribution
            st.markdown(f"""
            <div style="margin-bottom: 15px;">
                <div style="display: flex; height: 30px; width: 100%; border-radius: 4px; overflow: hidden; margin-bottom: 10px;">
                    <div style="flex: {len(bottom_regions)}; background-color: #ffcdd2; height: 100%;"></div>
                    <div style="flex: {len(lower_mid_regions)}; background-color: #fff9c4; height: 100%;"></div>
                    <div style="flex: {len(upper_mid_regions)}; background-color: #c8e6c9; height: 100%;"></div>
                    <div style="flex: {len(top_regions)}; background-color: #bbdefb; height: 100%;"></div>
                </div>
                <div style="display: flex; width: 100%; font-size: 12px; justify-content: space-between;">
                    <div style="text-align: left;">Low</div>
                    <div style="text-align: center;">← Productivity Distribution →</div>
                    <div style="text-align: right;">High</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create summary cards for each quartile
            st.markdown(f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 15px;">
                <div style="padding: 12px; background: #ffebee; border-radius: 6px; border-left: 4px solid #e53935;">
                    <div style="font-size: 14px; color: #555;">Bottom 25%</div>
                    <div style="font-size: 18px; font-weight: bold;">{round(bottom_regions['Productivity Score'].mean(), 1)}</div>
                    <div style="font-size: 12px; color: #777;">Avg. productivity</div>
                    <div style="margin-top: 5px; font-size: 12px;">{len(bottom_regions)} regions • {int(bottom_regions["Team Size"].sum())} workers</div>
                </div>
                <div style="padding: 12px; background: #fffde7; border-radius: 6px; border-left: 4px solid #fdd835;">
                    <div style="font-size: 14px; color: #555;">Lower Middle 25%</div>
                    <div style="font-size: 18px; font-weight: bold;">{round(lower_mid_regions['Productivity Score'].mean(), 1)}</div>
                    <div style="font-size: 14px; color: #555;">Lower Middle 25%</div>
                    <div style="font-size: 18px; font-weight: bold;">{round(lower_mid_regions['Productivity Score'].mean(), 1)}</div>
                    <div style="font-size: 12px; color: #777;">Avg. productivity</div>
                    <div style="margin-top: 5px; font-size: 12px;">{len(lower_mid_regions)} regions • {int(lower_mid_regions["Team Size"].sum())} workers</div>
                </div>
                <div style="padding: 12px; background: #e8f5e9; border-radius: 6px; border-left: 4px solid #43a047;">
                    <div style="font-size: 14px; color: #555;">Upper Middle 25%</div>
                    <div style="font-size: 18px; font-weight: bold;">{round(upper_mid_regions['Productivity Score'].mean(), 1)}</div>
                    <div style="font-size: 12px; color: #777;">Avg. productivity</div>
                    <div style="margin-top: 5px; font-size: 12px;">{len(upper_mid_regions)} regions • {int(upper_mid_regions["Team Size"].sum())} workers</div>
                </div>
                <div style="padding: 12px; background: #e3f2fd; border-radius: 6px; border-left: 4px solid #1e88e5;">
                    <div style="font-size: 14px; color: #555;">Top 25%</div>
                    <div style="font-size: 18px; font-weight: bold;">{round(top_regions['Productivity Score'].mean(), 1)}</div>
                    <div style="font-size: 12px; color: #777;">Avg. productivity</div>
                    <div style="margin-top: 5px; font-size: 12px;">{len(top_regions)} regions • {int(top_regions["Team Size"].sum())} workers</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Team size distribution analysis
            st.markdown("### Team Size Analysis")
            
            # Calculate average production per team size
            team_size_perf = best_assignments.groupby("Team Size")[["Predicted Quantity", "Productivity Score"]].mean().reset_index()
            
            # Display team size cards in a grid layout
            st.markdown(f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 15px;">
            """, unsafe_allow_html=True)
            
            # Generate cards for each team size
            for _, row in team_size_perf.iterrows():
                team_size = int(row["Team Size"])
                avg_pred = int(row["Predicted Quantity"])
                avg_prod = round(row["Productivity Score"], 1)
                team_count = len(best_assignments[best_assignments["Team Size"] == team_size])
                
                # Determine color based on productivity score
                if avg_prod >= productivity_quartiles[0.75]:
                    bg_color, border_color = "#e3f2fd", "#1e88e5"
                elif avg_prod >= productivity_quartiles[0.5]:
                    bg_color, border_color = "#e8f5e9", "#43a047"
                elif avg_prod >= productivity_quartiles[0.25]:
                    bg_color, border_color = "#fffde7", "#fdd835"
                else:
                    bg_color, border_color = "#ffebee", "#e53935"
                
                st.markdown(f"""
                <div style="padding: 12px; background: {bg_color}; border-radius: 6px; border-left: 4px solid {border_color};">
                    <div style="font-size: 14px; color: #555;">{team_size} Worker Teams</div>
                    <div style="font-size: 18px; font-weight: bold;">{avg_pred}</div>
                    <div style="font-size: 12px; color: #777;">Avg. production</div>
                    <div style="margin-top: 5px; font-size: 12px;">{team_count} teams • {avg_prod} productivity score</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display system performance metrics
        st.markdown('<div class="insight-section">', unsafe_allow_html=True)
        st.subheader("System Performance Metrics")
        
        # KPI row with 4 cards showing system performance
        cols = st.columns(4)
        
        # Worker efficiency KPI
        cols[0].markdown(f"""
        <div class="metric-card-1">
            <div class="metric-label">Worker Efficiency</div>
            <div class="metric-value">{kpis['worker_efficiency']}</div>
            <div>Units per worker</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Worker utilization KPI
        cols[1].markdown(f"""
        <div class="metric-card-2">
            <div class="metric-label">Worker Utilization</div>
            <div class="metric-value">{kpis['worker_utilization']}%</div>
            <div>Of available workers</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Warehouse coverage KPI
        cols[2].markdown(f"""
        <div class="metric-card-3">
            <div class="metric-label">Warehouse Coverage</div>
            <div class="metric-value">{kpis['coverage_ratio']}%</div>
            <div>Of regions covered</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Environmental score KPI
        cols[3].markdown(f"""
        <div class="metric-card-4">
            <div class="metric-label">Env. Optimization</div>
            <div class="metric-value">{kpis['env_optimization_score']}</div>
            <div>Working conditions score</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add completion time and processing metrics
        st.markdown(f"""
        <div style="display: flex; gap: 10px; margin-top: 20px;">
            <div style="flex: 2; padding: 15px; background: #f5f5f5; border-radius: 8px; border-left: 4px solid #757575;">
                <div style="font-size: 14px; color: #555;">Estimated Completion Time</div>
                <div style="font-size: 24px; font-weight: bold;">{kpis['est_completion_hours']} hours</div>
                <div style="font-size: 12px; color: #777;">Based on current assignments</div>
            </div>
            <div style="flex: 1; padding: 15px; background: #f5f5f5; border-radius: 8px; border-left: 4px solid #757575;">
                <div style="font-size: 14px; color: #555;">Processing Time</div>
                <div style="font-size: 24px; font-weight: bold;">{kpis['processing_time']} sec</div>
                <div style="font-size: 12px; color: #777;">Algorithm runtime</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display extrapolation warning if needed
        # if kpis['extrapolation_needed']:
        #     st.warning(f"Warning: Current temperature conditions are outside the model's training range. Predictions may be less reliable.")
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit app"""
    st.title("Warehouse Worker Assignment Tool")
    
    # Main content container
    with st.container():
        # st.markdown("""
        # <div class="card">
        # <p>This tool uses machine learning to optimize worker assignments based on historical performance data, 
        # current environmental conditions, and team composition. Enter the current warehouse conditions below to 
        # generate recommended worker team assignments.</p>
        # </div>
        # """, unsafe_allow_html=True)
        
        # Load and preprocess data
        df = load_data()
        df_filtered = preprocess_data(df)
        
        # Get unique hours from the dataset
        unique_hours = sorted(df_filtered["hour"].unique().tolist())
        
        # Get available regions
        regions = get_all_regions(df_filtered)
        
        # Input form for parameters
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Warehouse Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Temperature input with validation
            temp = st.number_input(
                "Temperature (°C)", 
                min_value=float(df_filtered["temperature"].min() - 5), 
                max_value=float(df_filtered["temperature"].max() + 5),
                value=22.0,
                step=0.5,
                help="Current warehouse temperature"
            )
            
        with col2:
            # Humidity input with validation
            humidity = st.number_input(
                "Humidity (%)", 
                min_value=10.0, 
                max_value=95.0,
                value=45.0,
                step=5.0,
                help="Current warehouse humidity"
            )
            
        with col3:
             regions = [1, 2, 3, 4, 5, 6]  # Define regions as integers from 1 to 6
             selected_region = st.selectbox(
             "Region",
             options=regions,
             index=0,
             help="Select a specific warehouse region (1-6)"
    )
        # Team size and number selection
        col1, col2 = st.columns(2)
        
        with col1:
            # Team size input
            team_size = st.slider(
                "Team Size", 
                min_value=2, 
                max_value=5,
                value=2,
                step=1,
                help="Number of workers per team"
            )
            
        with col2:
            # Top N teams selection
            top_n = st.slider(
                "Number of Top Teams", 
                min_value=1, 
                max_value=10,
                value=5,
                step=1,
                help="Number of top team combinations to show"
            )
            
        # Run model button
        run_button = st.button("Generate Worker Combination")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate and display recommendations
        if run_button:
            with st.spinner("Analyzing data and generating optimal worker assignments..."):
                # Show progress
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)  # Simulated processing time
                    progress.progress(i + 1)
                    
                # Get worker recommendations
                recommendations = recommend_worker_combinations(
                    df_filtered, 
                    temp, 
                    humidity, 
                    unique_hours,
                    selected_region=selected_region,
                    combination_size=team_size,
                    top_n=top_n
                )
                
                # Remove progress bar
                progress.empty()
                
                # Display results
                display_results(recommendations, temp, humidity, df_filtered, selected_region)
                
                # Add timestamp for when analysis was run
                st.info(f"Analysis completed at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        # else:
        #     # Display instructions when the tool first loads
        #     st.markdown("""
        #     <div class="optimization-card">
        #         <h3>How to Use This Tool</h3>
        #         <p>1. Enter the current warehouse temperature and humidity</p>
        #         <p>2. Select a specific region or "All" regions</p>
        #         <p>3. Choose your preferred team size (2-5 workers)</p>
        #         <p>4. Click "Generate Optimal Assignments" to run the analysis</p>
        #         <p>5. Review the results to see the optimal worker assignments for each region</p>
        #     </div>
        #     """, unsafe_allow_html=True)
            
        #     # Show tool capabilities
        #     st.markdown("""
        #     <div class="optimization-card">
        #         <h3>Key Features</h3>
        #         <ul>
        #             <li><b>Environmental Optimization:</b> Adjusts recommendations based on temperature and humidity</li>
        #             <li><b>Team Composition Analysis:</b> Finds optimal worker combinations for each region</li>
        #             <li><b>Performance Prediction:</b> Estimates productivity based on historical data</li>
        #             <li><b>Comparative Analytics:</b> Compares different team configurations</li>
        #             <li><b>Warehouse Coverage:</b> Ensures all regions have optimal worker assignments</li>
        #         </ul>
        #     </div>
           #

if __name__ == "__main__":
    script_path = os.path.abspath(sys.argv[0])  # Get the script path
    subprocess.run(["streamlit", "run", script_path], shell=True)
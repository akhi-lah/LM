import pandas as pd
import numpy as np
import pickle
import math
import random
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# 1. Data loading and preprocessing
# ---------------------------
def load_data(file_path="processed_warehouse_data.xlsx"):
    return pd.read_excel(file_path)

def preprocess_data(df):
    # Compute productivity rate (units per hour)
    df["productivity_rate"] = df["totalprocessedqty"] / (df["timediffseconds"] / 3600)
    # Interaction between temperature and humidity as an extra feature
    df["temp_hum_interaction"] = df["temperature"] * df["humidity"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["productivity_rate"], inplace=True)
    # Return three values to satisfy UI's unpacking (df, None, None)
    return df, None, None

# ---------------------------
# 2. Model training and evaluation
# ---------------------------
def train_model(df, model_filename="model.pkl"):
    features = [
        "temperature", "humidity", "estimatedpickdistancemeters", "estimatedstepstaken",
        "estimateditemweightkg", "totalweightprocessedkg", "productivity_rate", "temp_hum_interaction"
    ]
    scaler = MinMaxScaler()
    X = df[features]
    y = df[["totalprocessedqty", "timediffseconds"]]
 
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
 
    # Train two XGBoost regressors
    model_qty = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    model_time = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
 
    model_qty.fit(X_train_scaled, y_train["totalprocessedqty"])
    model_time.fit(X_train_scaled, y_train["timediffseconds"])
 
    # Save the models, scaler, and features
    with open(model_filename, "wb") as f:
        pickle.dump({
            "model_qty": model_qty,
            "model_time": model_time,
            "scaler": scaler,
            "features": features
        }, f)
 
def load_model(model_filename="model.pkl"):
    with open(model_filename, "rb") as f:
        return pickle.load(f)
 
# ---------------------------
# 3. Worker performance prediction
# ---------------------------
def predict_worker_performance(worker_data, model_data, scaler, features, temp, humidity):
    # Compute the mean feature vector for the worker
    worker_features = worker_data[features].mean()
    # Override temperature and humidity for prediction
    input_features = worker_features.to_dict()
    input_features["temperature"] = temp
    input_features["humidity"] = humidity
    input_features["temp_hum_interaction"] = temp * humidity
    input_df = pd.DataFrame([input_features], columns=features)
 
    input_scaled = scaler.transform(input_df)
    predicted_qty = model_data["model_qty"].predict(input_scaled)[0]
    predicted_time = model_data["model_time"].predict(input_scaled)[0]
 
    # Calculate productivity in units per hour
    productivity = predicted_qty / (predicted_time / 3600) if predicted_time != 0 else 0
    return predicted_qty, predicted_time, productivity
 
# ---------------------------
# 4. Assign workers to regions (zones) and prepare output for UI
# ---------------------------
def assign_workers_to_regions(region_df, worker_df, temp, humidity):
    model_data = load_model()
    features = model_data["features"]
    scaler = model_data["scaler"]
 
    unique_workers = worker_df["resource"].unique()
    worker_perf = {}
    avg_historical_productivity = worker_df["productivity_rate"].mean()
 
    for worker in unique_workers:
        w_data = worker_df[worker_df["resource"] == worker]
        pred_qty, pred_time, prod = predict_worker_performance(w_data, model_data, scaler, features, temp, humidity)
 
        # Ensure predictions are within reasonable bounds
        pred_qty = max(1.0, pred_qty)
        avg_worker_time = w_data["timediffseconds"].mean()
        min_time_threshold = avg_worker_time * 0.1
        pred_time = max(60.0, pred_time, min_time_threshold)
 
        # Recalculate productivity with adjusted time
        prod = pred_qty / (pred_time / 3600)
        if prod > 5 * avg_historical_productivity:
            prod = 5 * avg_historical_productivity
            pred_time = (pred_qty / prod) * 3600
 
        worker_perf[worker] = {
            "pred_qty": pred_qty,
            "pred_time": pred_time,
            "productivity": prod,
            "historical_avg_time": avg_worker_time
        }
 
    # Build region (zone) info based on quantity requirements
    regions = {}
    for idx, row in region_df.iterrows():
        region_id = row.get("region", row.get("zone"))
        quantity = max(1, row["quantity"])
 
        if quantity <= 50:
            target_workers = 1
        elif quantity <= 100:
            target_workers = 2
        elif quantity <= 300:
            target_workers = 3
        else:
            target_workers = 4
 
        regions[region_id] = {
            "required_qty": quantity,
            "assigned_workers": [],
            "target_workers": target_workers
        }
 
    total_required_workers = sum(data["target_workers"] for data in regions.values())
    total_available_workers = len(unique_workers)
    if total_required_workers > total_available_workers:
        deficit = total_required_workers - total_available_workers
        region_items = sorted(regions.items(), key=lambda x: x[1]["required_qty"])
        for i in range(deficit):
            if i < len(region_items):
                region_id, data = region_items[i]
                if data["target_workers"] > 1:
                    regions[region_id]["target_workers"] -= 1
 
    # Create a pool of available workers sorted by productivity (highest first)
    available_workers = list(worker_perf.keys())
    available_workers.sort(key=lambda w: worker_perf[w]["productivity"], reverse=True)
    sorted_regions = sorted(regions.keys(), key=lambda r: regions[r]["required_qty"], reverse=True)
 
    # Assign one worker per region first
    for region_id in sorted_regions:
        if available_workers:
            worker = available_workers.pop(0)
            regions[region_id]["assigned_workers"].append(worker)
 
    # Fill remaining assignments based on each region's target
    for region_id in sorted_regions:
        additional_needed = regions[region_id]["target_workers"] - len(regions[region_id]["assigned_workers"])
        for _ in range(additional_needed):
            if available_workers:
                worker = available_workers.pop(0)
                regions[region_id]["assigned_workers"].append(worker)
 
    # Prepare results for UI with individual details as a list of dictionaries
    results = []
    for region_id, data in regions.items():
        assigned = data["assigned_workers"]
        if not assigned:
            results.append({
                "Zone": region_id,
                "Processed_Quantity": data["required_qty"],
                "Team Size": 0,
                "EstimatedTimeToPickTheQuantity": "N/A",
                "Team Productivity": 0,
                "WorkerDetails": []
            })
            continue
 
        # --- Compute weighted team productivity ---
        # We'll weight each worker's productivity by their predicted quantity,
        # so that faster workers (with higher predicted qty) have more influence.
        worker_details = []
        total_weight = 0
        total_weighted_productivity = 0
        for worker in assigned:
            # Compute individual worker rate (for estimated time calculation)
            worker_rate = worker_perf[worker]["pred_qty"] / worker_perf[worker]["pred_time"]
            # Individual estimated time for region's required quantity:
            ind_time = data["required_qty"] / worker_rate
            hrs = int(ind_time // 3600)
            mins = int((ind_time % 3600) // 60)
            worker_details.append({
                "worker_id": worker,
                "Individual ETC": f"{hrs}hrs {mins}mins",
                "Individual Productivity": f"{round(worker_perf[worker]['productivity'], 2)} items/hr"
            })
            # Use predicted quantity as weight:
            weight = worker_perf[worker]["pred_qty"]
            total_weighted_productivity += worker_perf[worker]["productivity"] * weight
            total_weight += weight
 
        weighted_team_productivity = (total_weighted_productivity / total_weight) if total_weight > 0 else 0
 
        # Estimate team time for the region using the weighted team productivity:
        base_time_seconds = (data["required_qty"] / weighted_team_productivity) * 3600 if weighted_team_productivity > 0 else float('inf')
        scale_factor = 1.0 + (0.1 * math.log10(data["required_qty"])) if data["required_qty"] > 10 else 1.0
        est_time_seconds = max(60, base_time_seconds * scale_factor)
        hrs = int(est_time_seconds // 3600)
        mins = int((est_time_seconds % 3600) // 60)
        time_formatted = f"{hrs}hrs {mins}mins"
 
        results.append({
            "Zone": region_id,
            "Processed_Quantity": data["required_qty"],
            "Team Size": len(assigned),
            "EstimatedTimeToPickTheQuantity": time_formatted,
            "Team Productivity": round(weighted_team_productivity, 2),
            "WorkerDetails": worker_details
        })
 
    result_df = pd.DataFrame(results).sort_values("Zone")
    print("\nWorker assignment to regions:")
    print(result_df)
    return result_df
 
# ---------------------------
# 6. Main execution
# ---------------------------
if __name__ == "__main__":
    df = load_data()
    df, _, _ = preprocess_data(df)
    train_model(df)
 
    # Example region (zone) input
    region_input = pd.DataFrame({
        "region": [1, 2, 3, 4, 5, 6],
        "quantity": [50, 300, 500, 200, 700, 200]
    })
 
    temp = 12.0
    humidity = 60
    assign_workers_to_regions(region_input, df, temp, humidity)
